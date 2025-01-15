from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import beta
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY


def correct_for_point_bins(x, y):
    """
    Auxiliary function for reliably calibrated isotonic regression
    (train_rcir).
    Iff there is e.g. 2 samples that map to 0.5 and they have different
    classes but the exact same scores, then these will create a bin
    with zero width and only one entry in x and y for the isotonic regression
    and the interpolation model. Further functions don't know how to handle
    this, so we will correct for that in this function.
    """
    if(y[0] == y[1]):
        for i in range(int(len(y) / 2)):
            try:
                if(y[2 * i] == y[2 * i + 1]):
                    pass  # Everything in order.
                else:
                    # We found a bin represented by only _one_ entry in x and y. Fix that.
                    y = np.insert(y, 2 * i + 1, y[2 * i])
                    x = np.insert(x, 2 * i + 1, x[2 * i])
            except IndexError:
                pass  # Last element. Everything in order.
    elif(y[0] != y[1]):
        for i in range(int(len(y) / 2)):
            try:
                if(y[2 * i + 1] == y[2 * i + 2]):
                    pass
                else:
                    y = np.insert(y, 2 * i + 2, y[2 * i + 1])
                    x = np.insert(x, 2 * i + 2, x[2 * i + 1])
            except IndexError:
                pass  # Everything in order.
    return({'x': x, 'y': y})

def credible_interval(k, n, confidence_level=.95, tolerance=1e-6):
    """
    Auxiliary function for estimating width of credible interval.
    Finds the highest posterior density interval using binary search.
    
    Args:
    k (int): Number of positive samples
    n (int): Number of samples
    confidence_level (float): Probability mass that has to fall within
     the credible intervals. In ]0, 1].
    tolerance (float): Upper limit for tolerance of probability mass
     within credible interval.
    """
    p_min_lower = float(0)
    p_middle = p_min_upper = p_max_lower = k / float(n)
    p_max = p_max_upper = float(1)
    p_min_middle = (p_min_lower + p_middle) / 2  # == 0 if k == 0.
    p_max_middle = (p_middle + p_max) / 2  # == n if k == n
    if(k == 0):  # Exception handling
        # p_min_middle = 0  # Per definition... it's the peak.
        while(abs(beta.cdf(p_max_middle, 1, n + 1) - confidence_level) > tolerance):
            if(beta.cdf(p_max_middle, 1, n + 1) > confidence_level):
                p_max_upper = p_max_middle
            else:
                p_max_lower = p_max_middle
            p_max_middle = (p_max_lower + p_max_upper) / 2
    elif(k == n):  # Exception handling
        while(abs(1 - beta.cdf(p_min_middle, k + 1, 1) - confidence_level) > tolerance):
            if(1 - beta.cdf(p_min_middle, k + 1, 1) > confidence_level):
                p_min_lower = p_min_middle
            else:
                p_min_upper = p_min_middle
            p_min_middle = (p_min_lower + p_min_upper) / 2
    else:  # Main case
        while(abs(beta.cdf(p_max_middle, k + 1, n - k + 1) - beta.cdf(p_min_middle, k + 1, n - k + 1) -
                  confidence_level) > tolerance / 2):
            # Binary search
            # Reset p-max values for new iteration:
            p_max_lower = p_middle
            p_max_upper = p_max
            p_max_middle = (p_max_lower + p_max_upper) / 2
            while(abs(beta.logpdf(p_min_middle, k + 1, n - k + 1) -
                      beta.logpdf(p_max_middle, k + 1, n - k + 1)) > tolerance / 2):
                # Binary search to find p_max corresponding to p_min (same value in pdf).
                if(k * np.log(p_min_middle) + (n - k) * np.log(1 - p_min_middle) >
                   k * np.log(p_max_middle) + (n - k) * np.log(1 - p_max_middle)):
                    p_max_upper = p_max_middle
                else:
                    p_max_lower = p_max_middle
                p_max_middle = (p_max_lower + p_max_upper) / 2
            if(beta.cdf(p_max_middle, k + 1, n - k + 1) - beta.cdf(p_min_middle, k + 1, n - k + 1) >
               confidence_level):
                p_min_lower = p_min_middle
            else:
                p_min_upper = p_min_middle
            p_min_middle = (p_min_lower + p_min_upper) / 2
    return(dict([('p_min', p_min_middle), ('p_max', p_max_middle)]))

def predict(model, data_scores):
    try:  # IsotonicRegression model
        data_probabilities = model.predict(T=data_scores)
    except:
        try:  # Interpolation model
            data_probabilities = model(data_scores)
        except:  # kNN-model
            data_probabilities = model.predict_proba(X=data_scores)[:, 1]
    return(data_probabilities)

def estimate_performance(model, data_class, data_scores):
    """
    Function for estimating performance metrics (AUC-ROC and MSE)
    of model.
    
    Args:
    model {IsotonicRegression, interpolation omdel, kNN-model}:
     model used to convert scores to probabilities.
    data_class (np.array([])): Array of class labels. True indicates
     positive sample, False negative.
    data_scores (np.array([])): Scores produced e.g. by machine
     learning model for samples.
    """
    data_probabilities = predict(model, data_scores)
    # Estimate mean squared error:
    mse = sum((data_class - data_probabilities)**2) / len(data_class)
    # Estimate AUC-ROC:
    auc_roc = roc_auc_score(data_class, data_probabilities)
    res = dict([('mse', mse), ('auc_roc', auc_roc)])
    return(res)

def merge_bin(rcir_model, data_class, data_scores, merge_criterion='auc_roc'):
    """
    Auxiliary function for train_rcir. Performs one bin merge.
    Function could be hidden.
    """
    width_of_intervals = rcir_model['width of intervals']
    x = rcir_model['model'].x
    y = rcir_model['model'].y
    bin_summary = rcir_model['bin summary']
    credible_intervals = rcir_model['credible intervals']
    drop_idx = width_of_intervals.tolist().index(max(width_of_intervals))
    if drop_idx == 0:  # Exception handling. Fist bin has largest credible interval.
        # remove first two elements in x and y, update new first elements,
        # remove first element from width_of_intervals
        # and remove first elements from bin_summary[0] and bin_summary[1]
        y = np.delete(y, [0, 1])  # Drop first and second items.
        x = np.delete(x, [0, 1])
        new_prob = (bin_summary[0][0] * bin_summary[1][0] + bin_summary[0][1] * bin_summary[1][1]) / (bin_summary[1][0] + bin_summary[1][1])
        y[0] = new_prob
        try:  # y[1] doesn't exist if this is also the last bin.
            y[1] = new_prob
        except IndexError:
            pass
        # Leave x as is. bin_summary and width_of_intervals handled at end of loop.
        int_mod = interp1d(x, y, bounds_error=False)
        int_mod._fill_value_below = 0
        int_mod._fill_value_above = 1
        # print("Test-, and training performance, " + str(i) + " bins removed.d")
        # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
        # print(isotonic.estimate_performance(int_mod, training_class, training_scores))
        tmp = credible_interval(k=round(bin_summary[0][0] * bin_summary[1][0] + bin_summary[0][1] * bin_summary[1][1]),
                                n=(bin_summary[1][0] + bin_summary[1][1]))
        width_of_intervals[0] = tmp['p_max'] - tmp['p_min']
        credible_intervals.pop(drop_idx)  # Remove line from credible intervals
        credible_intervals[0] = tmp
        bin_summary[0][1] = new_prob
        bin_summary[1][1] = bin_summary[1][0] + bin_summary[1][1]
    elif drop_idx == len(width_of_intervals) - 1:
        # More exception handling. '-1' for last element?
        # remove last element (only one!) of x and y:
        two_y_end = False
        if(y[-1] == y[-2]):
            two_y_end = True
        y = np.delete(y, drop_idx * 2)
        y = np.delete(y, drop_idx * 2 - 1)
        x = np.delete(x, drop_idx * 2)
        x = np.delete(x, drop_idx * 2 - 1)
        new_prob = (bin_summary[0][-1] * bin_summary[1][-1] + bin_summary[0][-2] * bin_summary[1][-2]) / (bin_summary[1][-1] + bin_summary[1][-2])
        # Hmm, there might be two bins for this y
        if(two_y_end):
            y[-2] = new_prob
        y[-1] = new_prob
        tmp = credible_interval(k=round(bin_summary[0][-1] * bin_summary[1][-1] + bin_summary[0][-2] * bin_summary[1][-2]),
                                n=(bin_summary[1][-1] + bin_summary[1][-2]))
        width_of_intervals[-2] = tmp['p_max'] - tmp['p_min']
        credible_intervals.pop(-1)  # Drop last.
        credible_intervals[-1] = tmp
        bin_summary[0][-2] = new_prob
        bin_summary[1][-2] = bin_summary[1][-1] + bin_summary[1][-2]
        # if((drop_idx != len(width_of_intervals)) or (drop_idx != 0):  # Main handling
        int_mod = interp1d(x, y, bounds_error=False)
        int_mod._fill_value_below = 0
        int_mod._fill_value_above = 1
        # print("Testing set performance, " + str(i) + " bins removed.c")
        # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
    else:
        # Main method, i.e. when we are not dealing with the first or last bin.
        # y contains the probability to be dropped twice
        y = np.delete(y, drop_idx * 2 + 1)
        y = np.delete(y, drop_idx * 2)
        # Test lower:
        x_tmp_lower = np.array(x)  # Create NEW array!!
        x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2)  # Lower boundary of *this bin
        x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2 - 1)  # Upper boundary of smaller bin
        y_tmp_lower = np.array(y)  # Create _new_ array!!!
        new_prob_lower = ((bin_summary[1][drop_idx] * bin_summary[0][drop_idx] +
                          bin_summary[1][drop_idx - 1] * bin_summary[0][drop_idx - 1]) /
                          (bin_summary[1][drop_idx] + bin_summary[1][drop_idx - 1]))
        y_tmp_lower[drop_idx * 2 - 1] = new_prob_lower  # New value
        y_tmp_lower[drop_idx * 2 - 2] = new_prob_lower  # Same value
        # Test upper:
        x_tmp_upper = np.array(x)
        x_tmp_upper = np.delete(x, drop_idx * 2 + 2)  # Lower boundary of larger bin
        x_tmp_upper = np.delete(x_tmp_upper, drop_idx * 2 + 1)  # Upper boundary of *this bin
        y_tmp_upper = np.array(y)
        new_prob_upper = ((bin_summary[1][drop_idx] * bin_summary[0][drop_idx] +
                          bin_summary[1][drop_idx + 1] * bin_summary[0][drop_idx + 1]) /
                          (bin_summary[1][drop_idx] + bin_summary[1][drop_idx + 1]))
        y_tmp_upper[drop_idx * 2] = new_prob_upper  # New value, bin guaranteed to exist.
        try:  # Bin doesn't exist if it is last
            y_tmp_upper[drop_idx * 2 + 1] = new_prob_upper  # New value
        except IndexError:
            pass
        # Now, which bin to add it to?
        # Compare the two:
        int_mod_lower = interp1d(x_tmp_lower, y_tmp_lower, bounds_error=False)
        int_mod_upper = interp1d(x_tmp_upper, y_tmp_upper, bounds_error=False)
        int_mod_lower._fill_value_below = 0
        int_mod_lower._fill_value_above = 1
        int_mod_upper._fill_value_below = 0
        int_mod_upper._fill_value_above = 1
        # Left (smaller) bin: idx
        score_lower = estimate_performance(int_mod_lower, data_class, data_scores)
        score_upper = estimate_performance(int_mod_upper, data_class, data_scores)
        if((score_lower['auc_roc'] > score_upper['auc_roc'] and merge_criterion == 'auc_roc') or
           (score_lower['mse'] < score_upper['mse'] and merge_criterion == 'mse')):
            # Select the model with better auc_roc.
            x = x_tmp_lower
            y = y_tmp_lower
            bin_summary[1][drop_idx - 1] = bin_summary[1][drop_idx] + bin_summary[1][drop_idx - 1]
            bin_summary[0][drop_idx - 1] = new_prob_lower
            tmp = credible_interval(k=round(bin_summary[0][drop_idx - 1] * bin_summary[1][drop_idx - 1]),
                                    n=(bin_summary[1][drop_idx - 1]))
            width_of_intervals[drop_idx - 1] = tmp['p_max'] - tmp['p_min']
            credible_intervals.pop(drop_idx)
            credible_intervals[drop_idx - 1] = tmp
            int_mod = int_mod_lower
        else:
            x = x_tmp_upper
            y = y_tmp_upper
            bin_summary[1][drop_idx + 1] = bin_summary[1][drop_idx] + bin_summary[1][drop_idx + 1]
            bin_summary[0][drop_idx + 1] = new_prob_upper
            tmp = credible_interval(k=round(bin_summary[0][drop_idx + 1] * bin_summary[1][drop_idx + 1]),
                                    n=(bin_summary[1][drop_idx + 1]))
            width_of_intervals[drop_idx + 1] = tmp['p_max'] - tmp['p_min']
            credible_intervals[drop_idx + 1] = tmp
            credible_intervals.pop(drop_idx)
            int_mod = int_mod_upper
    width_of_intervals = np.delete(width_of_intervals, drop_idx)
    # Drop samples from bin_summary[1][drop_idx]. The samples are previously added to adequate new bin.
    bin_summary = (np.delete(bin_summary[0], drop_idx), np.delete(bin_summary[1], drop_idx))
    updated_rcir_model = {'model': int_mod, 'credible level': rcir_model['credible level'],
                          'credible intervals': credible_intervals, 'width of intervals': width_of_intervals,
                          'bin summary': bin_summary, 'd': rcir_model['d']}
    # Create new rcir_model object and return. (i.e. stich together new pieces of info.)
    return(updated_rcir_model)


@META_ARCH_REGISTRY.register()
class RCIR(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.best_model = None
        self.need_calibration_train = True
        self.require_iterative_training = False

    def forward(self, validation_logits, validation_labels, train_logits=None, train_labels=None):

        validation_logits = torch.from_numpy(validation_logits)
        validation_prob = F.softmax(validation_logits, dim=1)
        validation_scores,predictions = torch.max(validation_prob, 1)
        validation_scores = validation_scores.numpy()
        
        if self.training:
            train_logits = torch.from_numpy(train_logits)
            train_prob = F.softmax(train_logits, dim=1)
            train_scores,_ = torch.max(train_prob, 1)
            train_scores = train_scores.numpy()
            credible_level=.95
            y_min=0
            y_max=1
            merge_criterion='auc_roc'

            isotonic_regression_model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')
            isotonic_regression_model.fit(X=train_scores, y=train_labels)
            models = []
            # Extract the interpolation model we need:
            tmp_x = isotonic_regression_model.f_.x
            tmp_y = isotonic_regression_model.f_.y
            # Do some corrections (if there are any)
            tmp = correct_for_point_bins(tmp_x, tmp_y)
            x = tmp['x']
            y = tmp['y']
            # Use new boundaries to create an interpolation model that does the heavy lifting of
            # reliably calibrated isotonic regression:
            interpolation_model = interp1d(x=x, y=y, bounds_error=False)
            interpolation_model._fill_value_below = min(y)
            interpolation_model._fill_value_above = max(y)
            training_probabilities = interpolation_model(train_scores)
            # The following array contains all information defining the IR transformation
            bin_summary = np.unique(training_probabilities, return_counts=True)
            credible_intervals = [credible_interval(np.round(p * n), n) for (p, n) in
                                zip(bin_summary[0], bin_summary[1])]
            width_of_intervals = np.array([row['p_max'] - row['p_min'] for row in credible_intervals])
            rcir_model = {'model': interpolation_model, 'credible level': credible_level,
                        'credible intervals': credible_intervals, 'width of intervals': width_of_intervals,
                        'bin summary': bin_summary, 'd': -1}
            metrics = estimate_performance(rcir_model['model'], validation_labels, validation_scores)
            models.append([0, rcir_model['model'], metrics])
            while(len(rcir_model['width of intervals']) > 2):  # There still exists bins to merge
                rcir_model = merge_bin(rcir_model, train_labels, train_scores, merge_criterion)
                metrics = estimate_performance(rcir_model['model'], validation_labels, validation_scores)
                models.append([0, rcir_model['model'], metrics])
            best_model_idx = [item[2]['auc_roc'] for item in models].index(max([item[2]['auc_roc'] for item in models]))
            self.best_model = models[best_model_idx][1]
        else:
            cali_scores =  self.best_model(validation_scores)
            cali_scores = torch.from_numpy(cali_scores)
            validation_labels = torch.from_numpy(validation_labels)
            return cali_scores,predictions,validation_labels
        