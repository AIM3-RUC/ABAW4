import random
import torch
import numpy as np
import os
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.regressor import FcRegressor
from models.networks.classifier import FcClassifier
from models.networks.fft import FFTEncoder
from models.networks.transformer import TransformerEncoder

from models.loss import MSELoss, CELoss


class TransformerModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of transformer')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='transformer encoder hidden states')
        parser.add_argument('--num_layers', default=4, type=int, help='number of transformer encoder layers')
        parser.add_argument('--ffn_dim', default=1024, type=int, help='dimension of FFN layer of transformer encoder')
        parser.add_argument('--nhead', default=4, type=int, help='number of heads of transformer encoder')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='all', type=str, help='the pred target')
        parser.add_argument('--use_pe', action='store_true', help='whether to use position encoding')
        parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer', 'fft'], help='whether to use position encoding')
        parser.add_argument('--loss_type', type=str, default='multitask')
        parser.add_argument('--loss_weights', type=float, default=1, nargs='+')
        parser.add_argument('--weight_type', type=str, default='original',
                    choices=['original', 'weighted'])
        parser.add_argument('--save_model', default=False, action='store_true', help='whether to save_model at each epoch')
        parser.add_argument('--feature_layer_norm', default=False, action='store_true', help='whether to normalize feature before feeding to encoder')
        parser.add_argument('--dropout_all', default=False, action='store_true', help='whether to adjust dropout rate in encoder')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the Transformer class

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
        self.use_pe = opt.use_pe
        self.encoder_type = opt.encoder_type
        self.weight_type = opt.weight_type
        self.feature_layer_norm = opt.feature_layer_norm
        
        if self.feature_layer_norm:
            self.net_layer_norm = torch.nn.LayerNorm(opt.input_dim, eps=1e-6)
            self.model_names.append('_layer_norm')

        # net seq (already include a linear projection before the transformer encoder)
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        
        if self.encoder_type == 'transformer':
            if opt.dropout_all:
                self.net_seq = TransformerEncoder(opt.input_dim, opt.num_layers, opt.nhead, \
                                                dim_feedforward=opt.ffn_dim, affine=True, \
                                                affine_dim=opt.hidden_size, use_pe=self.use_pe, dropout = opt.dropout_rate)
            else:
                self.net_seq = TransformerEncoder(opt.input_dim, opt.num_layers, opt.nhead, \
                                                dim_feedforward=opt.ffn_dim, affine=True, \
                                                affine_dim=opt.hidden_size, use_pe=self.use_pe)
        elif self.encoder_type == 'fft':
            self.net_seq = FFTEncoder(opt.input_dim, opt.num_layers, opt.nhead,\
                                dim_feedforward=opt.ffn_dim, affine=True, affine_dim=opt.hidden_size)
        
        # net reg and cls
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.target_name = opt.target
        self.hidden_size = opt.hidden_size

        self.net_v_reg = FcRegressor(opt.hidden_size, layers, output_dim=1, dropout=opt.dropout_rate)
        self.net_a_reg = FcRegressor(opt.hidden_size, layers, output_dim=1, dropout=opt.dropout_rate)
        self.net_expr_cls = FcClassifier(opt.hidden_size, layers, output_dim=8, dropout=opt.dropout_rate)
        self.net_au_cls = FcClassifier(opt.hidden_size, layers, output_dim=12, dropout=opt.dropout_rate)
        
        # settings
        if self.isTrain:
            if self.isTrain:
                self.criterion_reg = MSELoss()
                self.criterion_cls = CELoss()
                self.criterio_bce = torch.nn.BCELoss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
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
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = {}
        self.feat = {}
        all_prediction = []
        all_feat = []
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction, feat = self.forward_step(feature_step, mask)
            all_prediction.append(dict([(key, value.detach()) for key, value in prediction.items()]))
            all_feat.append(dict([(key, value.detach()) for key, value in feat.items()]))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad() 
                
                target = {}
                target['v'] = self.v_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                target['a'] = self.a_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                target['expr'] = self.expr_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                target['au'] = self.au_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]

                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
                
        for key in all_prediction[0].keys():
            self.output[key] = []
        for prediction in all_prediction:
            for key, value in prediction.items():
                self.output[key].append(value.squeeze(dim=-1))
        for key in self.output.keys():
            self.output[key] = torch.cat(self.output[key], dim=1)
            
        for key in all_feat[0].keys():
            self.feat[key] = []
        for feat in all_feat:
            for key, value in feat.items():
                self.feat[key].append(value.squeeze(dim=-1))
        for key in self.feat.keys():
            self.feat[key] = torch.cat(self.feat[key], dim=1)
            
        
    def forward_step(self, input, mask):
        if self.feature_layer_norm:
            input = self.net_layer_norm(input)
        
        if self.encoder_type == 'fft':
            out, hidden_states = self.net_seq(input, mask) # hidden_states: layers * (seq_len, bs, hidden_size)
        else:
            out, hidden_states = self.net_seq(input) # hidden_states: layers * (seq_len, bs, hidden_size)
        last_hidden = hidden_states[-1].transpose(0, 1) # (bs, seq_len, hidden_size)
        
        # make prediction from last_hidden
        v_prediction, v_feat = self.net_v_reg(last_hidden)
        a_prediction, a_feat = self.net_a_reg(last_hidden)
        expr_logits, expr_feat = self.net_expr_cls(last_hidden)
        au_logits, au_feat = self.net_au_cls(last_hidden)
        
        return {'v':v_prediction, 'a':a_prediction, 'expr':expr_logits, 'au':au_logits}, {'v':v_feat, 'a':a_feat, 'expr':expr_feat, 'au':au_feat}
    
    def get_loss(self, v_loss=None, a_loss=None, au_loss=None, expr_loss=None):
        if self.weight_type == 'original':
            weights = (1, 1 ,1 ,1)
        elif self.weight_type == 'weighted':
            weights = (12, 12, 1, 0.35)
            
        losses = [v_loss, a_loss, au_loss, expr_loss]
        final_loss = 0
        for w, loss in zip(weights, losses):
            if loss is not None:
                final_loss = final_loss + w*loss

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
            if len(pred['au'][target['au']!=-1]) == 0:
                self.loss_au = 0
                pred['au'].detach_()
            else:
                _shape = pred['au'].shape
                self.loss_au = self.criterio_bce(F.sigmoid(pred['au'][target['au']!=-1]).view(-1, _shape[2]), target['au'][target['au']!=-1].view(-1, _shape[2]).float())
        
        ############ 对所有loss求和
        self.loss = self.get_loss(v_loss=self.loss_v, a_loss=self.loss_a, au_loss=self.loss_au, expr_loss=self.loss_expr)
        if 'au' in self.loss_type and len(pred['au'][target['au'] != -1]) == 0 and len(self.loss_type) == 1:
            pass
        else:
            self.loss.backward(retain_graph=False)
            
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
