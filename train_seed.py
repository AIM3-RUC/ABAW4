from ast import Raise
import os
import numpy as np, argparse, time, pickle, random, json

from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, evaluate_classification, remove_padding, scratch_data, smooth_func, metric_for_AU
from utils.tools import calc_total_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import fcntl
import csv
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def eval(model, val_iter, target_set, smooth=False):
    model.eval()
    total_pred = {}
    total_label = {}

    results = {}
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()

        pred, label = {}, {}
        pred['valence'] = model.output['v'].squeeze(-1).detach().cpu().numpy()
        pred['arousal'] = model.output['a'].squeeze(-1).detach().cpu().numpy()
        pred['expression'] = model.output['expr'].argmax(dim=-1).detach().cpu().numpy()
        pred['AU'] = F.sigmoid(model.output['au']).detach().cpu().numpy()
        pred['AU'] = np.int64(pred['AU']>=0.5)  # 以0.5为阈值进行分类

        label['valence'] = model.va_target[:,:,0].squeeze(-1).detach().cpu().numpy()
        label['arousal'] = model.va_target[:,:,1].squeeze(-1).detach().cpu().numpy()
        label['expression'] = model.expr_target.detach().cpu().numpy()
        label['AU'] = model.au_target.detach().cpu().numpy()

        for k, p in pred.items():
            if len(p.shape) != 1:
                pred[k] = remove_padding(pred[k], lengths)  # [size,
                label[k] = remove_padding(label[k], lengths)

            if total_pred.get(k) is None:
                total_pred[k] = []
                total_label[k] = []
            total_pred[k] += pred[k]
            total_label[k] += label[k]

        for j, video_id in enumerate(data['video_id']):
            for task in target_set:
                if results.get(task) is None:
                    results[task] = {}
                results[task][video_id] = {'video_id': video_id,
                                     'pred': pred[task][j].tolist(),
                                     'label': label[task][j].tolist()}

    # calculate metrics
    best_window = {}
    for task in target_set:
        window = None
        if smooth:
            total_pred[task], best_window[task] = smooth_func(total_pred[task], total_label[task],
                                                        best_window=window, logger=logger)

        total_pred[task] = scratch_data(total_pred[task])
        total_label[task] = scratch_data(total_label[task])
        assert len(total_pred[task]) == len(total_label[task])

    performance, confusion, all_f1 = {}, {}, {}
    for task in target_set:
        if task in ['valence', 'arousal']:
            total_pred[task] = total_pred[task][total_label[task]!=-5]
            total_label[task] = total_label[task][total_label[task]!=-5]
            _, _, _, performance[task] = evaluate_regression(total_label[task], total_pred[task])
        elif task in ['expression']:
            total_pred[task] = total_pred[task][total_label[task]!=-1]
            total_label[task] = total_label[task][total_label[task]!=-1]
            _, _, _, performance[task], confusion[task] = evaluate_classification(total_label[task], total_pred[task], average='macro')
            _, _, _, all_f1[task], _ = evaluate_classification(total_label[task], total_pred[task], average=None)

            expression_name_list = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
            all_f1[task] = dict(zip(expression_name_list, all_f1[task]))
        elif task in ['AU']:
            total_pred[task] = total_pred[task][total_label[task]!=-1]
            total_label[task] = total_label[task][total_label[task]!=-1]
            performance[task], all_f1[task] = metric_for_AU(total_label[task], total_pred[task])
        else:
            raise ValueError("Wrong eval target")

    model.train()

    return performance, all_f1 , results, confusion


def clean_chekpoints(checkpoints_dir, expr_name, store_epoch_list):
    root = os.path.join(checkpoints_dir, expr_name)
    # if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
    for checkpoint in os.listdir(root):
        isStoreEpoch = False
        for store_epoch in store_epoch_list:
            if checkpoint.startswith(str(store_epoch) + '_'):
                isStoreEpoch = True
                break
        if not isStoreEpoch and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


def write_avg_results(dir, best_eval_epoch, best_eval_perfomance, target_set):
    path = os.path.join(dir, 'results.txt')
    lines = {}
    if not os.path.exists(path):
        for target in target_set:
            lines[target] = ['%s :\tBest eval epoch %d with perfomance %f\n' % (target, best_eval_epoch[target], best_eval_perfomance[target])]
    else:
        with open(path, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                target = line.strip().split(' ')[0]
                if target == 'Average':
                    continue
                if not lines.get(target):
                    lines[target] = []
                lines[target].append(line)
        for target in target_set:
            lines[target] += ['%s :\tBest eval epoch %d with perfomance %f\n' % (target, best_eval_epoch[target], best_eval_perfomance[target])]

    all_lines = []
    for target in target_set:
        scores = []
        for line in lines[target]:
            scores.append(float(line.strip().split(' ')[-1]))
        average_score = sum(scores) / len(scores)
        lines[target] += ['Average Performance on task {} is {}\n\n'.format(target, round(average_score, 4))]
        all_lines += lines[target]
    with open(path, 'w') as f:
        f.writelines(all_lines)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    best_window = None
    opt = TrainOptions().parse()  # get training options

    seed = 99 + opt.run_idx
    seed_everything(seed)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    logger_path = os.path.join(opt.log_dir, opt.name)  # get logger path
    suffix = opt.name  # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    logger.info('Using seed: {}'.format(seed))

    if opt.full_data:
        set_name = ['trn_val', 'val']
    else:
        set_name = ['trn', 'val']
    dataset, val_dataset = create_dataset_with_args(opt, set_name=set_name)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    # calculate input dims
    if opt.feature_set != 'None':
        input_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.feature_set.split(','))))
        setattr(opt, "input_dim", input_dim)  # set input_dim attribute to opt
    if hasattr(opt, "a_features"):
        a_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.a_features.split(','))))
        setattr(opt, "a_dim", a_dim)  # set a_dim attribute to opt
    if hasattr(opt, "v_features"):
        v_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.v_features.split(','))))
        setattr(opt, "v_dim", v_dim)  # set v_dim attribute to opt
    if hasattr(opt, "l_features"):
        l_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.l_features.split(','))))
        setattr(opt, "l_dim", l_dim)  # set l_dim attribute to opt

    model = create_model(opt, logger=logger)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    if opt.target == 'all':
        target_set = ['valence', 'arousal', 'expression', 'AU']
    else:
        target_set_temp = opt.target.split(',')
        target_set = []
        if 'v' in target_set_temp:
            target_set.append('valence')
        if 'a' in target_set_temp:
            target_set.append('arousal')
        if 'expr' in target_set_temp:
            target_set.append('expression')
        if 'au' in target_set_temp:
            target_set.append('AU')
    
    # check the target_set
    for target in target_set:
        if target not in ['valence', 'arousal', 'expression', 'AU']:
            print(target_set)
            raise ValueError("Please check target")
    
    metrics = {}
    best_eval_perfomance = {}
    best_eval_epoch = {}
    best_eval_window = {}
    best_eval_result = {}
    
    for target in target_set:
        metrics[target] = []
        best_eval_perfomance[target] = 0
        best_eval_epoch[target] = -1
        best_eval_window[target] = None
        best_eval_result[target] = None
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        batch_count = 0  # batch个数统计，用于粗略计算整个epoch的loss
        iter_data_statis = 0.0  # record total data reading time
        cur_epoch_losses = OrderedDict()
        for name in model.loss_names:
            cur_epoch_losses[name] = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            iter_data_statis += iter_start_time - iter_data_time
            total_iters += 1  # opt.batch_size
            epoch_iter += opt.batch_size
            batch_count += 1
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.run()  # calculate loss functions, get gradients, update network weights

            # ---------在每个batch都获取一次loss，并加入cur_epoch_losses-------------
            losses = model.get_current_losses()
            for name in losses.keys():
                cur_epoch_losses[name] += losses[name]
            # ---------------------------------------------------------------------

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec, Data loading: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, iter_data_statis))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

        # -----得到并打印当前epoch的loss------
        for name in cur_epoch_losses:
            cur_epoch_losses[name] /= batch_count
        logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                    ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**cur_epoch_losses))
        # -----------------------------------

        # ---tensorboard---
        writer = SummaryWriter(os.path.join(logger_path, 'tensorboard'))
        for name in cur_epoch_losses:
           writer.add_scalar("Loss_{}/train".format(name), cur_epoch_losses[name], epoch)
        # -----------------

        # eval train set
        for train, div_dataset in zip([True, False], [dataset, val_dataset]):
            perfomances, all_f1s, preds, confusions = eval(model, div_dataset, target_set)
            for target in target_set:
                perfomance, pred = perfomances[target], preds[target]
                all_f1, cm = all_f1s.get(target), confusions.get(target)
                if train:
                    logger.info('%s:\tTrn result of epoch %d / %d \tperformance %.4f' % (target, epoch, opt.niter + opt.niter_decay, perfomance))
                else:
                    logger.info('%s:\tVal result of epoch %d / %d \tperformance %.4f' % (
                    target, epoch, opt.niter + opt.niter_decay, perfomance))
                if target in ['expression']:
                    logger.info(cm)
                    logger.info(all_f1)
                if target in ['AU']:
                    logger.info(all_f1)

                if not train:
                    metrics[target].append((perfomance))
                    if perfomance > best_eval_perfomance[target]:
                        best_eval_epoch[target] = epoch
                        best_eval_perfomance[target] = perfomance
                        best_eval_window[target] = None
                        best_eval_result[target] = pr
    
    best_epoch_list = []
    for target in target_set:
        logger.info('Best eval epoch %d found with perfomance %f on %s' % (best_eval_epoch[target], best_eval_perfomance[target], target))
        logger.info(opt.name)
        result_save_path = os.path.join(opt.log_dir, opt.name, '{}-epoch-{}_preds.json'.format(target, best_eval_epoch[target]))
        json.dump(best_eval_result[target], open(result_save_path, 'w'))
        best_epoch_list.append(best_eval_epoch[target])
        
        if opt.full_data:
            if target == 'valence':
                save_epoch_list = [i for i in range(10,20)]
            elif target == 'arousal':
                save_epoch_list = [i for i in range(15,20)]
            elif target == 'expression':
                save_epoch_list = [i for i in range(15,25)]
            elif target == 'AU':
                save_epoch_list = [i for i in range(30,35)]
            best_epoch_list.extend(save_epoch_list)
        
    clean_chekpoints(opt.checkpoints_dir, opt.name, best_epoch_list)
    write_avg_results(os.path.dirname(logger_path) if len(logger_path.split('/')) > 1 else logger_path,
                      best_eval_epoch, best_eval_perfomance, target_set)
