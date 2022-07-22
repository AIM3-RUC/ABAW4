import torch
import numpy as np
import os
import torch.nn.functional as F
import sys
from utils.tools import calc_total_dim
from models.base_model import BaseModel
from models.loss import MSELoss, CELoss
from models.networks.classifier import FcClassifier
from models.networks.regressor import FcRegressor
from models.networks.lstm_encoder import LSTMEncoder, BiLSTMEncoder


class LstmModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--bidirection', default=False, action='store_true',
                            help='whether to use bidirectional lstm')
        parser.add_argument('--loss_type', type=str, default='multitask')
        parser.add_argument('--cls_weighted', default=False, action='store_true', help='whether to use weighted cls')
        parser.add_argument('--weight_type', type=str, default='original',
                            choices=['original', 'weighted'])
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)

        self.loss_type = opt.loss_type.split(',')
        if 'multitask' in self.loss_type:
            self.loss_names = ['v', 'a', 'expr', 'au']
        else:
            for loss in self.loss_type:
                if loss not in ['v', 'a', 'expr', 'au']:
                    raise ValueError("Please check loss_type")
            self.loss_names = self.loss_type

        self.model_names = ['_seq', '_v_reg', '_a_reg', '_expr_cls', '_au_cls']
        self.pretrained_model = []
        self.max_seq_len = opt.max_seq_len
        self.weight_type = opt.weight_type
        # net seq
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        if opt.bidirection:
            self.net_seq = BiLSTMEncoder(opt.input_dim, opt.hidden_size)
            self.hidden_mul = 2
        else:
            self.net_seq = LSTMEncoder(opt.input_dim, opt.hidden_size)
            self.hidden_mul = 1

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_v_reg = FcRegressor(opt.hidden_size * self.hidden_mul, layers, output_dim=1, dropout=opt.dropout_rate)
        self.net_a_reg = FcRegressor(opt.hidden_size * self.hidden_mul, layers, output_dim=1, dropout=opt.dropout_rate)
        self.net_expr_cls = FcClassifier(opt.hidden_size * self.hidden_mul, layers, output_dim=8, dropout=opt.dropout_rate)
        self.net_au_cls = FcClassifier(opt.hidden_size * self.hidden_mul, layers, output_dim=12, dropout=opt.dropout_rate)
        # settings FcRegressor(opt.hidden_size * self.hidden_mul, layers, 1, dropout=opt.dropout_rate)
        self.target_name = opt.target
        if self.isTrain:
            if self.isTrain:
                self.criterion_reg = MSELoss()
                self.criterion_cls = CELoss()
                # self.criterion_cls = FocalLoss()
                self.criterio_bce = torch.nn.BCELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.feature = input['feature'].to(self.device)
        self.mask = input['mask'].to(self.device)

        self.length = input['length']
        if load_label:
            self.va_target = torch.stack([input['valence'], input['arousal']], dim=2).to(self.device)
            self.v_target = input['valence'].to(self.device)
            self.a_target = input['arousal'].to(self.device)
            self.expr_target = input['expression'].to(self.device)

            AU_name_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
            self.au_target = torch.stack([input[AU_name] for AU_name in AU_name_list], dim=2).to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = {}
        all_prediction = []
        previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        for step in range(split_seg_num):
            feature_step = self.feature[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]
            prediction, (previous_h, previous_c) = self.forward_step(feature_step, (previous_h, previous_c))
            previous_h = previous_h.detach()
            previous_c = previous_c.detach()
            all_prediction.append(dict([(key, value.detach()) for key, value in prediction.items()]))
            # backward
            if self.isTrain:
                mask = self.mask[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]
                self.optimizer.zero_grad()

                target = {}
                target['v'] = self.v_target[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]
                target['a'] = self.a_target[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]
                target['expr'] = self.expr_target[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]
                target['au'] = self.au_target[:, step * self.max_seq_len: (step + 1) * self.max_seq_len]

                self.backward_step(prediction, target, mask)
                self.optimizer.step()

        for key in all_prediction[0].keys():
            self.output[key] = []
        for prediction in all_prediction:
            for key, value in prediction.items():
                self.output[key].append(value.squeeze(dim=-1))
        for key in self.output.keys():
            self.output[key] = torch.cat(self.output[key], dim=1)

    def forward_step(self, input, states):
        hidden, (h, c) = self.net_seq(input, states)
        v_prediction, _ = self.net_v_reg(hidden)
        a_prediction, _ = self.net_a_reg(hidden)
        expr_logits, _ = self.net_expr_cls(hidden)
        au_logits, _ = self.net_au_cls(hidden)

        return {'v': v_prediction, 'a': a_prediction, 'expr': expr_logits, 'au': au_logits}, (h, c)

    def get_loss(self, v_loss=None, a_loss=None, au_loss=None, expr_loss=None):
        if self.weight_type == 'original':
            weights = (1, 1, 1, 1)
        elif self.weight_type == 'weighted':
            weights = (12, 12, 1, 0.35)

        losses = [v_loss, a_loss, au_loss, expr_loss]
        final_loss = 0
        for w, loss in zip(weights, losses):
            if loss is not None:
                final_loss = final_loss + w * loss

        return final_loss

    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        self.loss_v, self.loss_a, self.loss_au, self.loss_expr = None, None, None, None
        ############ valence loss
        if 'v' in self.loss_names:
            v_mask = mask.unsqueeze(-1)  # -> [B, L, 1]
            target_mask = target['v'] != -5
            target_mask = target_mask.int().unsqueeze(-1)
            v_mask = v_mask.mul(target_mask)

            self.loss_v = self.criterion_reg(pred['v'], target['v'].unsqueeze(-1), v_mask)

        ############ arousal loss
        if 'a' in self.loss_names:
            a_mask = mask.unsqueeze(-1)  # -> [B, L, 1]
            target_mask = target['a'] != -5
            target_mask = target_mask.int().unsqueeze(-1)
            a_mask = a_mask.mul(target_mask)
            self.loss_a = self.criterion_reg(pred['a'], target['a'].unsqueeze(-1), a_mask)

        ############ expr loss
        if 'expr' in self.loss_names:
            expr_mask = mask
            expr_pred = pred['expr'].permute(0, 2, 1)
            self.loss_expr = self.criterion_cls(expr_pred, target['expr'], expr_mask)

        ############ au loss
        if 'au' in self.loss_names:
            if len(pred['au'][target['au'] != -1]) == 0:
                self.loss_au = 0
                pred['au'].detach_()
            else:
                _shape = pred['au'].shape
                self.loss_au = self.criterio_bce(F.sigmoid(pred['au'][target['au'] != -1]).view(-1, _shape[2]),
                                                 target['au'][target['au'] != -1].view(-1, _shape[2]).float())

        ############ 对所有loss求和
        self.loss = self.get_loss(v_loss=self.loss_v, a_loss=self.loss_a, au_loss=self.loss_au,
                                  expr_loss=self.loss_expr)
        if 'au' in self.loss_type and len(pred['au'][target['au'] != -1]) == 0 and len(self.loss_type) == 1:
            pass
        else:
            self.loss.backward(retain_graph=False)

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 5)
