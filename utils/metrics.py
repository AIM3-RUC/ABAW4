import time
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, precision_score
import warnings
warnings.filterwarnings("ignore")


def remove_padding(batch_data, lengths):
    ans = []
    for i in range(batch_data.shape[0]):
        ans.append(batch_data[i, :lengths[i]])
    return ans

def scratch_data(data_lst):
    data = np.concatenate(data_lst)
    return data

def smooth_predictions(preds, window=40, mean_or_binomial=True):
    """
    Args:
      preds: list of (subj_timesteps, dim), len(list)=num_subjs
      window: int (one side timesteps to average)
    Return:
      smoothed_preds: shape=preds
    """
    if window == 0:
        return preds
    if not mean_or_binomial:
        from scipy.stats import binom
        binomial_weights = np.zeros((window * 2 + 1,))
        for i in range(window + 1):
            binomial_weights[i] = binomial_weights[-i - 1] = binom.pmf(i, window * 2, 0.5)
    smoothed_preds = []
    for pred in preds:
        smoothed_pred = []
        for t in range(len(pred)):
            left = np.max([0, t - window])
            right = np.min([len(pred), t + window])
            if mean_or_binomial:
                smoothed_pred.append(np.mean(pred[left: right + 1]))
            else:
                if left <= 0:
                    weights = binomial_weights[window - t:]
                elif right >= len(pred):
                    weights = binomial_weights[:len(pred) - t - window - 1]
                else:
                    weights = binomial_weights
                smoothed_pred.append(np.sum(pred[left: right + 1] * weights))
        smoothed_preds.append(np.array(smoothed_pred))
    smoothed_preds = np.array(smoothed_preds)
    return smoothed_preds

def smooth_func(pred, label=None, best_window=None, logger=None):
    start = time.time()
    if best_window is None:
        best_ccc, best_window = 0, 0
        for window in range(0, 20, 5):
            smoothed_preds = smooth_predictions(pred, window=window)
            if label is not None:
                mse, rmse, pcc, ccc = evaluate_regression(y_true=scratch_data(label),
                                                        y_pred=scratch_data(smoothed_preds))
                if logger:
                    logger.info('In smooth Eval \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}'.format(
                        window, mse, rmse, pcc, ccc))
                else:
                    pass
                
                if ccc > best_ccc:
                    best_ccc, best_window = ccc, window
        end = time.time()
        time_usage = end - start
        if logger:
            logger.info('Smooth: best window {:d} best_ccc {:.4f} \t Time Taken {:.4f}'.format(best_window, best_ccc, time_usage))
        else:
            print('Smooth: best window {:d} best_ccc {:.4f} \t Time Taken {:.4f}'.format(best_window, best_ccc, time_usage))
        smoothed_preds = smooth_predictions(pred, window=best_window)
    elif best_window is not None:
        smoothed_preds = smooth_predictions(pred, window=best_window)
    
    return smoothed_preds, best_window
    
def evaluate_regression(y_true, y_pred):
    """ Evaluate the regression performance
        Params:
        y_true, y_pred: np.array()
        Returns:
        mse, rmse, pcc, ccc
    """
    assert y_true.ndim==1 and y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_true, y_pred)[0][1]
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    ccc = 2 * np.cov(y_true, y_pred, ddof=0)[0][1] / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return mse, rmse, pcc, ccc

def evaluate_classification(y_true, y_pred, average='macro'):
    '''
    Evaluate the classification performance
        Params:
        y_true, y_pred: np.array()
        Returns:
        acc, recall, precision, f1, cm
    '''
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    # recall_for_each_class = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=average) 
    # preision_for_each_class = precision_score(y_true, y_pred, average=None) 
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    return acc, recall, precision, f1, cm

def metric_for_AU(y_true, y_pred):
    assert len(y_true)%12 == 0
    assert len(y_true) == len(y_pred)
    
    y_true = y_true.reshape(int(len(y_true)/12), 12)
    y_pred = y_pred.reshape(int(len(y_pred)/12), 12)
    
    all_f1 = []
    
    for t in range(12):
        y_true_ = y_true[:, t]
        y_pred_ = y_pred[:, t]
        all_f1.append(f1_score(y_true_, y_pred_))
    
    f1_mean = np.mean(all_f1)
    
    AU_name_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
    all_f1 = dict(zip(AU_name_list, all_f1))
    
    return f1_mean, all_f1