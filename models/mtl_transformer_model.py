import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.networks.regressor import FcRegressor
from models.networks.classifier import FcClassifier
from models.networks.fft import FFTEncoder
from models.networks.transformer import TransformerEncoder
from models.loss import MSELoss, CELoss

class MTLTransformerModel(BaseModel):
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
        parser.add_argument('--loss_weights', type=str, default='12,12,1,0.35')
        parser.add_argument('--strategy', default='original', type=str,
                            choices=['original', 'feedback', 'feature_feedback'], help='the strategy of multi-task learning')
        parser.add_argument('--structure', default='original', type=str,
                            choices=['original', 'share_bottom', 'only_classifier'], help='the structure of multi-task learning')
        parser.add_argument('--save_model', default=False, action='store_true', help='whether to save_model at each epoch')
        parser.add_argument('--weight_type', type=str, default='fixed',
                            choices=['fixed', 'normalized_fixed', 'param_adjust', 'loss_adjust'])
        parser.add_argument('--feed_source', type=str, default=['au'], nargs='+', choices=['au', 'v', 'a', 'expr'])
        parser.add_argument('--feed_target', type=str, default=['expr'], nargs='+', choices=['au', 'v', 'a', 'expr'])
        parser.add_argument('--feed_gt_rate', type=float, default=0)
        parser.add_argument('--check_feed_metrics', default=False, action='store_true')
        parser.add_argument('--gt_rate_decay', default=False, action='store_true')
        parser.add_argument('--feed_dim', type=int, default=256)
        parser.add_argument('--share_layers', type=int, default=1)
        parser.add_argument('--cheating', type=bool, default=False)
        return parser

    @staticmethod
    def get_encoder(input_dim, num_layers, nhead, dim_feedforward, affine, affine_dim, use_pe, dropout):
        return TransformerEncoder(input_dim, num_layers, nhead, dim_feedforward=dim_feedforward,
                                  affine=affine, affine_dim=affine_dim, use_pe=use_pe, dropout=dropout)

    def __init__(self, opt, logger=None):
        """Initialize the Transformer class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        
        self.loss_type = opt.loss_type.split(',')
        self.opt = opt
        self.strategy = opt.strategy
        self.structure = opt.structure
        if 'multitask' in self.loss_type:
            self.loss_names = ['v', 'a', 'expr', 'au']
        else:
            for loss in self.loss_type:
                if loss not in ['v', 'a', 'expr', 'au']:
                    raise ValueError("Please check loss_type")
            self.loss_names = self.loss_type
        self.task_losses = dict([(name, []) for name in self.loss_names])
        
        self.model_names = ['_seq', '_v_reg', '_a_reg', '_expr_cls', '_au_cls']
        self.pretrained_model = []
        self.max_seq_len = opt.max_seq_len
        self.use_pe = opt.use_pe
        self.encoder_type = opt.encoder_type

        # net seq (already include a linear projection before the transformer encoder)
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        if opt.structure == 'share_bottom':
            opt.num_layers = opt.num_layers - opt.share_layers

        self.no_encoder = False
        if opt.num_layers <= 0 or opt.structure == 'only_classifier':
            self.no_encoder = True
        if self.no_encoder:
            self.model_names.remove('_seq')
            opt.hidden_size = opt.input_dim
        else:
            if self.encoder_type == 'transformer':
                self.net_seq = self.get_encoder(opt.input_dim, opt.num_layers, opt.nhead,
                                                dim_feedforward=opt.ffn_dim, affine=True,
                                                affine_dim=opt.hidden_size, use_pe=self.use_pe, dropout=opt.dropout_rate)
            elif self.encoder_type == 'fft':
                self.net_seq = FFTEncoder(opt.input_dim, opt.num_layers, opt.nhead,
                                    dim_feedforward=opt.ffn_dim, affine=True, affine_dim=opt.hidden_size)
        
        # net reg and cls
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.target_name = opt.target
        self.hidden_size = opt.hidden_size

        task_hidden_size = dict([(t, opt.hidden_size) for t in ['au', 'v', 'a', 'expr']])
        if opt.strategy == 'feedback':
            if 'au' in opt.feed_source:
                self.model_names += ['_au_embedding']
                self.net_au_embedding = nn.Embedding(12, opt.feed_dim)
            if 'v' in opt.feed_source:
                self.model_names += ['_v_embedding']
                self.net_v_embedding = nn.Embedding(1, opt.feed_dim)
            if 'a' in opt.feed_source:
                self.model_names += ['_a_embedding']
                self.net_a_embedding = nn.Embedding(1, opt.feed_dim)
            if 'expr' in opt.feed_source:
                self.model_names += ['_expr_embedding']
                self.net_expr_embedding = nn.Embedding(8, opt.feed_dim)

        if opt.strategy in ['feedback', 'feature_feedback']:
            for task in opt.feed_target:
                for _ in opt.feed_source:
                    task_hidden_size[task] += opt.feed_dim if opt.strategy == 'feedback' else layers[-1]

        v_layers = [FcRegressor(task_hidden_size['v'], layers, output_dim=1, dropout=opt.dropout_rate)]
        a_layers = [FcRegressor(task_hidden_size['a'], layers, output_dim=1, dropout=opt.dropout_rate)]
        expr_layers = [FcClassifier(task_hidden_size['expr'], layers, output_dim=8, dropout=opt.dropout_rate)]
        au_layers = [FcClassifier(task_hidden_size['au'], layers, output_dim=12, dropout=opt.dropout_rate)]

        if self.structure == 'share_bottom':
            v_layers = [self.get_encoder(task_hidden_size['v'], opt.share_layers, task_hidden_size['v'] // 64,
                                          dim_feedforward=opt.ffn_dim, affine=False, affine_dim=task_hidden_size['v'],
                                          use_pe=self.use_pe, dropout=opt.dropout_rate)] + v_layers
            a_layers = [self.get_encoder(task_hidden_size['a'], opt.share_layers, task_hidden_size['a'] // 64,
                                          dim_feedforward=opt.ffn_dim, affine=False, affine_dim=task_hidden_size['a'],
                                          use_pe=self.use_pe, dropout=opt.dropout_rate)] + a_layers
            expr_layers = [self.get_encoder(task_hidden_size['expr'], opt.share_layers, task_hidden_size['expr'] // 64,
                                            dim_feedforward=opt.ffn_dim, affine=False, affine_dim=task_hidden_size['expr'],
                                            use_pe=self.use_pe, dropout=opt.dropout_rate)] + expr_layers
            au_layers = [self.get_encoder(task_hidden_size['au'], opt.share_layers, task_hidden_size['au'] // 64,
                                          dim_feedforward=opt.ffn_dim, affine=False, affine_dim=task_hidden_size['au'],
                                          use_pe=self.use_pe, dropout=opt.dropout_rate)] + au_layers

        self.net_v_reg = nn.ModuleList(v_layers)
        self.net_a_reg = nn.ModuleList(a_layers)
        self.net_expr_cls = nn.ModuleList(expr_layers)
        self.net_au_cls = nn.ModuleList(au_layers)
        self.weight_type = opt.weight_type
        if self.weight_type == 'param_adjust':
            self.net_weight = nn.Parameter(torch.randn(4))

        self.loss_weights = [float(w) for w in opt.loss_weights.split(',')]
          
        # settings
        if self.isTrain:
            self.criterion_reg = MSELoss()
            self.criterion_cls = CELoss()
            self.criterio_bce = torch.nn.BCELoss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            if self.weight_type == 'param_adjust':
                paremeters += [{'params': self.net_weight}]
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
        all_prediction = []
        for step in range(split_seg_num):
            target = {}
            target['v'] = self.va_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len, 0].unsqueeze(-1)
            target['a'] = self.va_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len, 1].unsqueeze(-1)
            target['expr'] = self.expr_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            target['au'] = self.au_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]

            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction = self.forward_step(feature_step, mask, target=target)
            all_prediction.append(prediction)
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()
                self.backward_step(prediction, target, mask)
                self.optimizer.step()
        for key in all_prediction[0].keys():
            self.output[key] = []
        for prediction in all_prediction:
            for key, value in prediction.items():
                self.output[key].append(value.squeeze(dim=-1))
        for key in self.output.keys():
            self.output[key] = torch.cat(self.output[key], dim=1)

    def predict(self, hidden, task='v', return_hidden=False):
        if task == 'v':
            module = self.net_v_reg
        if task == 'a':
            module = self.net_a_reg
        elif task == 'au':
            module = self.net_au_cls
        elif task == 'expr':
            module = self.net_expr_cls

        if isinstance(module, torch.nn.DataParallel):
            module = module.module

        if self.structure == 'share_bottom':
            predictions, last_hidden = module[1](module[0](hidden)[1][-1].transpose(0, 1))
        else:
            predictions, last_hidden = module[0](hidden)

        if return_hidden:
            return predictions, last_hidden
        else:
            return predictions

    def forward_step(self, input, mask, target=None):
        if self.no_encoder:
            last_hidden = input
        else:
            if self.encoder_type == 'fft':
                out, hidden_states = self.net_seq(input, mask) # hidden_states: layers * (seq_len, bs, hidden_size)
            else:
                out, hidden_states = self.net_seq(input) # hidden_states: layers * (seq_len, bs, hidden_size)
            last_hidden = hidden_states[-1].transpose(0, 1) # (bs, seq_len, hidden_size)

        v_hidden, a_hidden, au_hidden, expr_hidden = last_hidden, last_hidden, last_hidden, last_hidden

        logits, task_last_hiddens = {}, {}
        if self.strategy in ['feedback', 'feature_feedback']:
            gt = {'v': target['v'],
                  'a': target['a'],
                  'au': target['au'],
                  'expr': torch.zeros(target['expr'].shape + (8,)).cuda()}
            for i in range(8):
                if torch.sum(target['expr'] == i) != 0:
                    temp = gt['expr'][target['expr'] == i]
                    temp[..., i] = 1 #torch的实现有毒
                    gt['expr'][target['expr'] == i] = temp

            for task in ['v', 'a', 'au', 'expr']:
                if task not in self.opt.feed_target:
                    logits[task], task_last_hiddens[task] = self.predict(last_hidden, task, return_hidden=True)

            for task in self.opt.feed_target:
                features = [last_hidden]
                for t in self.opt.feed_source:
                    #if self.opt.cheating:
                    if self.strategy == 'feature_feedback':
                        features.append(task_last_hiddens[t].detach())
                    elif self.isTrain or self.opt.cheating:
                        gt_rate = self.opt.feed_gt_rate
                        if self.opt.gt_rate_decay:
                            all_epochs = self.opt.niter + self.opt.niter_decay
                            gt_rate = gt_rate * (all_epochs - self.epoch) * 1.0 / all_epochs
                        features.append(self.get_task_emb(t, logits[t].detach(), gt[t], gt_rate))
                    else:
                        features.append(self.get_task_emb(t, logits[t].detach()))
                features = torch.cat(features, dim=-1)
                logits[task] = self.predict(features, task)

        for hidden, task in zip([v_hidden, a_hidden, au_hidden, expr_hidden], ['v', 'a', 'au', 'expr']):
            if task not in logits.keys():
                logits[task] = self.predict(hidden, task)

        for task in logits.keys():
            if task not in self.loss_names:
                logits[task].detach_()
        return logits

    def get_task_emb(self, task, pred, gt=None, gt_rate=0):
        if pred is not None and task == 'expr':
            pred = torch.softmax(pred, dim=-1)
        elif pred is not None and task == 'au':
            pred = torch.sigmoid(pred)

        if gt is None:
            assert gt_rate == 0
            gt = torch.zeros_like(pred)
        if pred is None:
            assert gt_rate == 1 and gt is not None
            pred = torch.zeros_like(gt)

        length = pred.shape[1]
        perm = torch.randperm(length).cuda()
        gt_length, pred_length = int(length * gt_rate), length-int(length * gt_rate)
        temp_weights = torch.zeros_like(pred)

        if gt_length != 0:
            temp_weights[:, perm[:gt_length]] = gt[:, perm[:gt_length]].float()
        if pred_length != 0:
            temp_weights[:, perm[gt_length:]] = pred[:, perm[gt_length:]].float()

        if task in ['v', 'a']:
            temp = torch.LongTensor([0]).cuda()
            if task == 'v':
                embedding = self.net_v_embedding(temp) #[1, dim]
            else:
                embedding = self.net_a_embedding(temp)
            weights = torch.zeros_like(pred)
            weights[temp_weights != -5] = (temp_weights[temp_weights != -5] + 1) / 2 #[-1, 1] -> [0, 1]
            features = torch.sum(embedding.unsqueeze(0).unsqueeze(1) * weights.unsqueeze(-1), dim=2) #[B, L, dim]
        elif task == 'expr':
            temp = torch.LongTensor([i for i in range(8)]).cuda()
            embedding = self.net_expr_embedding(temp) #[8, dim]
            weights = temp_weights
            features = torch.sum(embedding.unsqueeze(0).unsqueeze(1) * weights.unsqueeze(-1), dim=2)
        elif task == 'au':
            temp = torch.LongTensor([i for i in range(12)]).cuda()
            embedding = self.net_au_embedding(temp)  # [12, dim]
            weights = torch.zeros_like(pred)
            weights[temp_weights != -1] = temp_weights[temp_weights != -1]  #[B, L, 12]
            features = torch.sum(embedding.unsqueeze(0).unsqueeze(1) * weights.unsqueeze(-1), dim=2)

        return features #[B, L, dim]

    def get_weights(self, default=(15, 15, 1, 0.25), temperature=1.5):
        if self.weight_type in ['fixed', 'normalized_fixed']:
            return default
        elif self.weight_type == 'param_adjust':
            weights = []
            for j, (name, weight) in enumerate(zip(['v', 'a', 'au', 'expr'], self.net_weight)):
                if name in self.loss_names:
                    weights.append(default[j] / torch.square(weight))
                else:
                    weights.append(default[j])
            return weights
        elif self.weight_type == 'loss_adjust':
            ratio = []
            ratio_sum = 0
            for j, name in enumerate(['v', 'a', 'au', 'expr']):
                if name in self.loss_names and len(self.task_losses[name]) > 1:
                    ratio.append(self.task_losses[name][-1] / self.task_losses[name][-2] / temperature)
                    ratio_sum += math.exp(self.task_losses[name][-1] / self.task_losses[name][-2] / temperature)
                else:
                    ratio.append(1 / temperature)

            weights = []
            for j, name in enumerate(['v', 'a', 'au', 'expr']):
                if name in self.loss_names and len(self.task_losses[name]) > 1:
                    weight = len(self.loss_names) * math.exp(ratio[j]) / ratio_sum * default[j]
                    weights.append(weight)
                else:
                    weights.append(default[j])
            return weights

    def get_loss(self, v_loss=None, a_loss=None, au_loss=None, expr_loss=None, pred=None, target=None):
        weights = self.get_weights(self.loss_weights)
        losses = [v_loss, a_loss, au_loss, expr_loss]
        final_loss = 0
        for w, loss in zip(weights, losses):
            if loss is not None:
                if self.weight_type not in ['fixed', 'param_adjust'] and type(loss) != int and loss > 0: #除了fixed，其他默认都normalize到1
                    loss = loss/loss.item()
                final_loss = final_loss + w*loss

        if self.weight_type == 'param_adjust':
            for name, weight in zip(['v', 'a', 'au', 'expr'], self.net_weight):
                if name in self.loss_names:
                    final_loss = final_loss + torch.log(weight)

        return final_loss
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        ############ v and a loss
        self.loss_v, self.loss_a, self.loss_au, self.loss_expr = None, None, None, None
        if 'v' in self.loss_names:
            v_mask = mask.unsqueeze(-1)  # -> [B, L, 1]
            target_mask = target['v'] != -5
            target_mask = target_mask.int()
            v_mask = v_mask.mul(target_mask)
            self.loss_v = self.criterion_reg(pred['v'], target['v'], v_mask)

        if 'a' in self.loss_names:
            a_mask = mask.unsqueeze(-1)  # -> [B, L, 1]
            target_mask = target['a'] != -5
            target_mask = target_mask.int()
            a_mask = a_mask.mul(target_mask)
            self.loss_a = self.criterion_reg(pred['a'], target['a'], a_mask)
        
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
                self.loss_au = self.criterio_bce(F.sigmoid(pred['au'][target['au']!=-1]), target['au'][target['au']!=-1].float())
        
        ############ 对所有loss求和
        if 'au' in self.loss_type and len(pred['au'][target['au'] != -1]) == 0:
            self.loss = self.get_loss(v_loss=self.loss_v, a_loss=self.loss_a, au_loss=None, expr_loss=self.loss_expr,
                                      pred=pred, target=target)
        else:
            self.loss = self.get_loss(v_loss=self.loss_v, a_loss=self.loss_a, au_loss=self.loss_au,
                                      expr_loss=self.loss_expr, pred=pred, target=target)
            
        if type(self.loss) != int:
            self.loss.backward(retain_graph=False)
            
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)