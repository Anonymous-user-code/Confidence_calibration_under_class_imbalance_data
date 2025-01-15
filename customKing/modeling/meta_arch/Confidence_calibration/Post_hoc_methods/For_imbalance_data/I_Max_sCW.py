'''
Reference paper: "MULTI-CLASS UNCERTAINTY CALIBRATION VIA MUTUAL INFORMATION MAXIMIZATION-BASED BINNING"
Reference code: https://github.com/boschresearch/imax-calibration
'''


import io
import numpy as np
from tqdm import tqdm
import scipy
import scipy.integrate as integrate
from sklearn.utils.extmath import row_norms, stable_cumsum
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from sklearn.preprocessing import OneHotEncoder
import deepdish
import torch

class BaseCalibrator():
    """
    A generic base class.
    """

    def __init__(self):
        self.parameter_list = []

    def fit(self, logits, logodds, y, **kwargs):
        """
        Function to learn the model parameters using the input data X and labels y.

        Parameters
        ----------
        logits: numpy ndarray
            input data to the calibrator.
        logodds: numpy ndarray
            input data to the calibrator.
        y: numpy ndarray
            target labels
        Returns
        -------

        """
        raise NotImplementedError("Subclass must implement this method.")

    def calibrate(self, logits, logodds, **kwargs):
        """
        Calibrate the data using the learned parameters after fit was already called.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)

    def save_params(self, fpath):
        """
        Save the parameters of the model. The parameters which need to be saved are determined by self.parameter_list.
        Saves a single hdf5 file with keys being the parameter names.

        Parameters
        ----------
        fpath: string
            filepath to save the hdf5 file with model parameters
        Returns
        -------
        """
        if len(self.parameter_list) > 0:
            data_to_save = {}
            for key in self.parameter_list:
                data_to_save[key] = getattr(self, key)
            io.deepdish_write(fpath, data_to_save)
            print(("Parameters written to fpath: %s" % (fpath)))

    def load_params(self, fpath):
        """
        Load the parameters of the model. The parameters which need to be loaded are determined by self.parameter_list.
        Loads a single hdf5 file and assigns the attributes to the object using keys as the parameter names.

        Parameters
        ----------
        fpath: string
            filepath to save the hdf5 file with model parameters
        Returns
        -------
        """
        if len(self.parameter_list) > 0:
            data_to_load = io.deepdish_read(fpath)
            for key in self.parameter_list:
                setattr(self, key, data_to_load[key])
            print(("Parameters loaded and updated from fpath: %s" % (fpath)))

def binary_convertor(logodds, y, cal_setting, class_idx):
    """
    Function to convert the logodds data (in multi-class setting) to binary setting. The following options are available:
    1) CW - slice out some class: cal_setting="CW" and class_idx is not None (int)
    2) top1 - max class for each sample: get the top1 prediction: cal_setting="top1" and class_idx is None
    3) sCW - merge marginal setting where data is combined: cal_setting="sCW" and class_idx is None
    """

    if cal_setting == "CW":
        assert class_idx is not None, "class_idx needs to be an integer to slice out class needed for CW calibration setting"
        logodds_c = logodds[..., class_idx]
        y_c = y[..., class_idx] if y is not None else None
    elif cal_setting == "top1":
        assert class_idx is None, "class_idx needs to be None - check"
        top1_indices = logodds.argmax(axis=-1)
        logodds_c = logodds[np.arange(top1_indices.shape[0]), top1_indices]
        y_c = y.argmax(axis=-1) == top1_indices if y is not None else None
    elif cal_setting == "sCW":
        assert class_idx is None, "class_idx needs to be None - check"
        logodds_c = np.concatenate(logodds.T)
        y_c = np.concatenate(y.T) if y is not None else None
    else:
        raise Exception("Calibration setting (%s) not recognized!" % (cal_setting))

    return logodds_c, y_c

def fit_kde_distributions(logodds, y):
    """
    Fit KDEs to the data based on the labels. Get KDE-pos and KDE-neg.

    Parameters
    ----------
    logoddds: numpy ndarray
        logodds (lambda)
    y: numpy ndarray
        binary labels indicating positive or negative label

    Returns
    -------
    dict:
        KDE dictionary with "pos" and "neg"
    """
    distr_pos = scipy.stats.gaussian_kde(logodds[y == 1])
    distr_neg = scipy.stats.gaussian_kde(logodds[y == 0])
    return {"pos": distr_pos, "neg": distr_neg}

def bin_data(x, bins):
    """
    Given bin boundaries quantize the data (x). When ndims(x)>1 it will flatten the data, quantize and then reshape back to orig shape.
    Returns the following quantized values for num_bins=10 and bins = [2.5, 5.0, 7.5, 1.0]\n
    quantize: \n
              (-inf, 2.5) -> 0\n
              [2.5, 5.0) -> 1\n
              [5.0, 7.5) -> 2\n
              [7.5, 1.0) -> 3\n
              [1.0, inf) -> 4\n

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array
    bins: numpy ndarray
        location of the (num_bins-1) bin boundaries

    Returns
    -------
    assigned: int numpy ndarray
        For each sample, this contains the bin id (0-indexed) to which the sample belongs.
    """
    orig_shape = x.shape
    # if not 1D data. so need to reshape data, then quantize, then reshape back
    if len(orig_shape) > 1 or orig_shape[-1] != 1:
        x = x.flatten()
    assigned = np.digitize(x, bins)  # bin each input in data. np.digitize will always return a valid index between 0 and num_bins-1 whenever bins has length (num_bins-1) to cater for the open range on both sides
    if len(orig_shape) > 1 or orig_shape[-1] != 1:
        assigned = np.reshape(assigned, orig_shape)
    return assigned.astype(np.int64)

def safe_log_diff(A, B, log_func=np.log):
    """
    Numerically stable log difference function. Avoids log(0). Will compute log(A/B) safely where the log is determined by the log_func
    """
    EPS = np.finfo(float).eps
    if np.isscalar(A):
        if A == 0 and B == 0:
            return log_func(EPS)
        elif A == 0:
            return log_func(EPS) - log_func(B)
        elif B == 0:
            return log_func(A) - log_func(EPS)
        else:
            return log_func(A) - log_func(B)
    else:
        # log(A) - log(B)
        output = np.where(A == 0, log_func(EPS), log_func(A)) - np.where(B == 0, log_func(EPS), log_func(B))
        output[np.logical_or(A == 0, B == 0)] = log_func(EPS)
        assert np.all(np.isfinite(output))
        return output

def to_logodds(x):
    """

    Convert probabilities to logodds using:

    .. math::
        \\log\\frac{p}{1-p} ~ \\text{where} ~ p \\in [0,1]

    Natural log.

    Parameters
    ----------
    x : numpy ndarray
       Class probabilties as numpy array.

    Returns
    -------
    logodds : numpy ndarray
       Logodds output

    """
    assert x.max() <= 1 and x.min() >= 0
    numerator = x
    denominator = 1-x
    #numerator[numerator==0] = EPS
    # denominator[denominator==0] = EPS # 1-EPS is basically 1 so not stable!
    logodds = safe_log_diff(numerator, denominator, np.log)  # logodds = np.log( numerator/denominator   )
    assert np.all(np.isfinite(logodds)) == True, "Logodds output contains NaNs. Handle this."
    return logodds

def bin_repr_unknown_LLR(sample_weights, assigned, num_bins, return_probs=False):
    """
    Unknown Bin reprs. Will take the average of the either the pred_probs or the binary labels.
    Determines the bin reprs by taking average of sample weights in each bin.
    For example for sample-based repr: sample_weights should be 0 or 1 indicating correctly classified or not.
    or for pred-probs-based repr: sample_weights should be the softmax output probabilities.
    Handles reshaping if sample_weights or assigned has more than 1 dim.

    Parameters
    ----------
    sample_weights: numpy ndarray
        array with the weight of each sample. These weights are used to calculate the bin representation by taking the averages of samples grouped together.
    assigned: int numpy array
        array with the bin ids of each sample
    return_probs: boolean (default: True)
        All operations take place in logodds space. Setting this to true will ensure that the values returned are in probability space (i.e. it will convert the quantized values from logodds to sigmoid before returning them)

    Returns
    -------
    representations: numpy ndarray
        representations of each sample based on the bin it was assigned to
    """
    orig_shape = sample_weights.shape
    assert np.all(orig_shape == assigned.shape)
    assert sample_weights.max() <= 1.0 and sample_weights.min() >= 0.0, "make sure sample weights are probabilities"
    if len(orig_shape) > 1:
        sample_weights = sample_weights.flatten()
        assigned = assigned.flatten()

    bin_sums_pos = np.bincount(assigned, weights=sample_weights, minlength=num_bins)  # sum up all positive samples
    counts = np.bincount(assigned, minlength=num_bins)  # sum up all samples in bin
    filt = counts > 0
    prob_pos = np.ones(num_bins)*sample_weights.mean()  # NOTE: important change: when no samples at all fall into any bin then default should be the prior
    prob_pos[filt] = bin_sums_pos[filt] / counts[filt]  # get safe prob of pos samples over all samples
    representations = prob_pos
    if return_probs == False:
        representations = to_logodds(representations)  # NOTE: converting to logit domain again
    return representations

def to_sigmoid(x):
    """
    Stable sigmoid in numpy. Uses tanh for a more stable sigmoid function.

    Parameters
    ----------
    x : numpy ndarray
       Logits of the network as numpy array.

    Returns
    -------
    sigmoid: numpy ndarray
       Sigmoid output
    """
    sigmoid = 0.5 + 0.5 * np.tanh(x/2)
    assert np.all(np.isfinite(sigmoid)) == True, "Sigmoid output contains NaNs. Handle this."
    return sigmoid

def bin_representation_calculation(x, y, num_bins, bin_repr_scheme="sample_based", bin_boundaries=None, assigned=None, return_probs=False):
    """
    Bin representations: frequency based: num_positive_samples/num_total_samples in each bin.
        or pred_prob based: average of the sigmoid of lambda
    Function gets the bin representation which can be used during the MI maximization.

    Parameters
    ----------
    x: numpy ndarray
        logodds data which needs to be binned using bin_boundaries. Only needed if assigned not given.
    y: numpy ndarray
        Binary label for each sample
    bin_repr_scheme: strig
        scheme to use to determine bin reprs. options: 'sample_based' and 'pred_prob_based'
    bin_boundaries: numpy array
        logodds bin boundaries. Only needed when assigned is not given.
    assigned: int numpy array
        bin id assignments for each sample

    Returns
    -------
    quant_reprs: numpy array
        quantized bin reprs for each sample

    """
    assert (bin_boundaries is None) != (assigned is None), "Cant have or not have both arguments. Need exactly one of them."
    if assigned is None:
        assigned = bin_data(x, bin_boundaries)

    if bin_repr_scheme == "sample_based":
        quant_reprs = bin_repr_unknown_LLR(y, assigned, num_bins, return_probs)  # frequency estimate of correct/incorrect
    elif bin_repr_scheme == "pred_prob_based":
        quant_reprs = bin_repr_unknown_LLR(to_sigmoid(x), assigned, num_bins, return_probs)  # softmax probability for bin reprs
    else:
        raise Exception("bin_repr_scheme=%s is not valid." % (bin_repr_scheme))
    return quant_reprs

def bin_representation_function(logodds, labels, num_bins, bin_repr_scheme="sample_based"):
    """
    Get function which returns the sample based bin representations. The function will take in bin boundaries as well as the logodds and labels to return the representations.

    Parameters
    ----------
    logodds: numpy ndarray
        validation logodds
    labels: numpy logodds
        binary labels

    Returns
    -------
    get_bin_reprs: function
        returns a function which takes in bin_boundaries

    """
    def get_bin_reprs(bin_boundaries):
        return bin_representation_calculation(logodds, labels, num_bins, bin_repr_scheme, bin_boundaries=bin_boundaries)
    return get_bin_reprs

def bin_boundary_update_closed_form(representations):
    """
    Closed form update of boundaries. stationary point when log(p(y=1|lambda)) - log(p(y=0|lambda)) = log(log(xxx)/log(xxx)) term. LHS side is logodds/boundaries when p(y|lambda) modelled with sigmoid (e.g. PPB )
    """
    EPS = np.finfo(float).eps
    temp_log = 1. + np.exp(-1*np.abs(representations))
    temp_log[temp_log == 0] = EPS
    logphi_a = np.maximum(0., representations) + np.log(temp_log)
    logphi_b = np.maximum(0., -1*representations) + np.log(temp_log)
    assert np.any(np.sign(logphi_a[1:]-logphi_a[:-1])*np.sign(logphi_b[:-1]-logphi_b[1:]) >= 0.)
    temp_log1 = np.abs(logphi_a[1:] - logphi_a[:-1])
    temp_log2 = np.abs(logphi_b[:-1] - logphi_b[1:])
    temp_log1[temp_log1 == 0] = EPS
    temp_log2[temp_log2 == 0] = EPS
    bin_boundaries = np.log(temp_log1) - np.log(temp_log2)
    bin_boundaries = np.sort(bin_boundaries)
    return bin_boundaries

def bin_boundary_function():
    def get_bin_boundary(representations):
        return bin_boundary_update_closed_form(representations)
    return get_bin_boundary

def MI_known_LLR(bin_boundaries, p_y_pos, distr_kde_dict):
    """
    Calculate the MI(lambda_hat, y)(using the known LLR), where lambda_hat is the quantized lambdas.
    This will compute the MI in bits (log2).
    It uses a KDE to estimate the density of the positive and negative samples.
    At the end it will perform some basic checks to see if the computations were correct.
    In addition to the MI it will compute the bit rate (R) (i.e. MI(z, lambda) where z is quantized lambda)


    Parameters
    ----------
    bin_boundaries: numpy array
        bin boundaries
    p_y_pos: float
        p(y=1) prior
    distr_kde_dict: dict
        dictionary containing the KDE objects used to estimate the density in each bin with keys 'pos' and 'neg'.

    Returns
    -------
    MI: float
        MI(z, y) where z is quantized lambda. This is the mutual information between the quantizer output to the label.
    R: float
        bin rate. This is MI(z, lambda). Mutual Information between lambda and quantized lambda.
    """
    EPS = np.finfo(float).eps
    distr_pos, distr_neg = distr_kde_dict["pos"], distr_kde_dict["neg"]
    p_y_neg = 1 - p_y_pos

    new_boundaries = np.hstack([-100, bin_boundaries, 100])
    # lists for checks afterwards
    all_vs, all_intpos, all_intneg = [], [], []
    MI, R = 0.0, 0.0
    for idx in range(len(bin_boundaries) + 1):
        integral_pos = p_y_pos*distr_pos.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1])  # p(lam|y=1)*p(y=1) = p(lam|y=1)
        integral_neg = p_y_neg*distr_neg.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1])  # p(lam|y=1)*p(y=1) = p(lam|y=0)
        repr = safe_log_diff(integral_pos, integral_neg, np.log)

        p_ypos_given_z = max(EPS, to_sigmoid(repr))
        p_yneg_given_z = max(EPS, to_sigmoid(-1*repr))

        curr_MI_pos = integral_pos * (safe_log_diff(p_ypos_given_z, p_y_pos, np.log2))
        curr_MI_neg = integral_neg * (safe_log_diff(p_yneg_given_z, p_y_neg, np.log2))
        MI += curr_MI_pos + curr_MI_neg

        v = max(EPS, (integral_pos + integral_neg))
        curr_R = -1 * v * np.log2(v)  # entropy of p(z) = p(z|y=1)p(y=1) + p(z|y=0)p(y=0)
        R += curr_R
        # gather for checks
        all_vs.append(v)
        all_intpos.append(integral_pos)
        all_intneg.append(integral_neg)
    np.testing.assert_almost_equal(np.sum(all_vs), 1.0, decimal=2)
    np.testing.assert_almost_equal(np.sum(all_intpos), p_y_pos, decimal=2)
    np.testing.assert_almost_equal(np.sum(all_intneg), p_y_neg, decimal=2)
    return MI, R

def mutual_information_and_R_function(p_y_pos, distr_kde_dict):
    '''logodds_c => the logodds which were used to bin. rewrote MI loss: sum_Y sum_B p(y'|lambda)p(lambda) for term outside log. Before it was p(lambda|y')p(y') '''
    # NOTE: checked and matches impl of Dan: -1*MI_eval(**kwargs) => all good
    def get_MI(bin_boundaries):
        return MI_known_LLR(bin_boundaries, p_y_pos, distr_kde_dict)
    return get_MI

def MI_unknown_LLR(p_y_pos, logodds, bin_boundaries, representations):
    """logodds => the logodds which were used to bin. rewrote MI loss: sum_Y sum_B p(y'|lambda)p(lambda) for term outside log. Before it was p(lambda|y')p(y') """
    # NOTE: checked and matches impl of Dan: -1*MI_eval(**kwargs) => all good
    pred_probs = to_sigmoid(logodds)
    prior_y = dict(pos=p_y_pos, neg=1-p_y_pos)
    num_bins = len(bin_boundaries)+1
    # get p(y|lambda)p(lambda).... first get mean pred. prob. per bin
    assigned = bin_data(logodds, bin_boundaries)
    bin_sums_pred_probs_pos = np.bincount(assigned, weights=pred_probs, minlength=num_bins)  # get the reprs in prob space because of mean.
    p_y_pos_given_lambda_per_bin = bin_sums_pred_probs_pos / logodds.shape[0]
    bin_sums_pred_probs_neg = np.bincount(assigned, weights=1-pred_probs, minlength=num_bins)  # get the reprs in prob space because of mean.
    p_y_neg_given_lambda_per_bin = bin_sums_pred_probs_neg / logodds.shape[0]
    p_y_given_lambda_dict = dict(pos=p_y_pos_given_lambda_per_bin, neg=p_y_neg_given_lambda_per_bin)
    mi_loss = 0.0
    for binary_class_str, binary_class in zip(["neg", "pos"], [0, 1]):
        terms_in_log = (1 + np.exp((1-2*binary_class) * representations)) * prior_y[binary_class_str]   # part 3
        bin_summation_term = np.sum(p_y_given_lambda_dict[binary_class_str] * np.log(terms_in_log))
        mi_loss += bin_summation_term
    return -1*mi_loss

def unknown_LLR_mutual_information(p_y_pos, logodds):
    def get_MI(bin_boundaries, representations):
        return MI_unknown_LLR(p_y_pos, logodds, bin_boundaries, representations)
    return get_MI

def MI_upper_bounds(p_y_pos, distr_kde_dict):
    """
    Calculate the MI upper bound of MI(z, y) <= MI(lambda, y). As z is the quantized version of lambda, MI(z, y) is upper bounded by MI(lambda, y).
    This is a tigther bound than H(y). This function will return both upper bounds.

    Bound 1: MI(z, y) <= H(y) - H(y|z) <= H(y)
    Bound 2: MI(z, y) <= MI(lambda, y)

    Parameters
    ----------
    p_y_pos: float
        p(y=1) prior
    distr_kde_dict: dict
        dictionary containing the KDE objects used to estimate the density in each bin with keys 'pos' and 'neg'.

    Returns
    -------
    H_y: float
        Loose upper bound which is H(y)
    MI_y_lambda: float
        Upper bound of MI(z, y) which is upper bounded by MI(lambda, y). Tigther bound than H(y)

    """
    p_y_neg = 1 - p_y_pos

    # Bound 1
    H_y = -1*p_y_pos*np.log2(p_y_pos) + -1*p_y_neg*np.log2(p_y_neg)

    # Bound 2
    distr_pos, distr_neg = distr_kde_dict["pos"], distr_kde_dict["neg"]

    def get_logodd_lambda(lam):
        log_term_1 = p_y_pos * distr_pos.pdf(lam)
        log_term_2 = p_y_neg * distr_neg.pdf(lam)
        logodd_lambda = safe_log_diff(log_term_1, log_term_2, np.log)
        return logodd_lambda

    def integral_pos(lam):
        logodd_lambda = get_logodd_lambda(lam)
        p_ypos_lambda = to_sigmoid(logodd_lambda)
        return p_y_pos * distr_pos.pdf(lam) * safe_log_diff(p_ypos_lambda, p_y_pos, np.log2)  # np.log2(  p_ypos_lambda    /   p_y_pos  )

    def integral_neg(lam):
        logodd_lambda = get_logodd_lambda(lam)
        p_yneg_lambda = to_sigmoid(-1 * logodd_lambda)
        return p_y_neg * distr_neg.pdf(lam) * safe_log_diff(p_yneg_lambda, p_y_neg, np.log2)  # np.log2(  p_yneg_lambda    /   p_y_neg  )

    term_pos = integrate.quad(integral_pos, -100, 100, limit=100)[0]
    term_neg = integrate.quad(integral_neg, -100, 100, limit=100)[0]
    MI_y_lambda = term_pos + term_neg
    return H_y, MI_y_lambda

def CE_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    CE_mtx = - logits_q * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q)))
    return CE_mtx


def KL_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    KL_mtx = (logits_p - logits_q) * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q))) - np.maximum(0., logits_p) - np.log(1. + np.exp(-abs(logits_p)))
    #KL_mtx = - logits_q * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q)))
    return KL_mtx


def JSD_mtx(logits_p, logits_q):
    logits_p_a = np.reshape(logits_p.astype(np.float64), [logits_p.shape[0], 1])
    logits_q_a = np.reshape(logits_q.astype(np.float64), [1, logits_q.shape[0]])
    logits_q_a = logits_q_a * 0.5 + 0.5 * logits_p_a
    KL_mtx_a = (logits_p_a - logits_q_a) * (0.5 + 0.5*np.tanh(logits_p_a/2.)) + np.maximum(0., logits_q_a) + \
        np.log(1. + np.exp(-abs(logits_q_a))) - np.maximum(0., logits_p_a) - np.log(1. + np.exp(-abs(logits_p_a)))

    logits_p_b = np.reshape(logits_p.astype(np.float64), [1, logits_p.shape[0]])
    logits_q_b = np.reshape(logits_q.astype(np.float64), [logits_q.shape[0], 1])
    logits_p_b = logits_q_b * 0.5 + 0.5 * logits_p_b
    KL_mtx_b = (logits_q_b - logits_p_b) * (0.5 + 0.5*np.tanh(logits_q_b/2.)) + np.maximum(0., logits_p_b) + \
        np.log(1. + np.exp(-abs(logits_p_b))) - np.maximum(0., logits_q_b) - np.log(1. + np.exp(-abs(logits_q_b)))
    return KL_mtx_a * 0.5 + KL_mtx_b.transpose()*0.5

def kmeans_pp_init(X, n_clusters, random_state, n_local_trials=None, mode='jsd'):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape
    random_state = np.random.RandomState(random_state)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    center_ids = np.empty((n_clusters,), dtype=np.int64)

    #assert x_squared_norms is not None, 'x_squared_norms None in _k_init'
    x_squared_norms = row_norms(X, squared=True)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    #test_id   = random_state.randint(n_samples)
    # assert test_id != center_id:
    center_ids[0] = center_id
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    if mode == 'euclidean':
        closest_dist_sq = euclidean_distances(centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True)
    elif mode == 'kl':
        # def KL_div(logits_p, logits_q):
        #    assert logits_p.shape[1] == 1 or logits_q.shape[1] == 1
        #    return (logits_p - logits_q) * (np.tanh(logits_p/2.) * 0.5 + 0.5) + np.maximum(logits_q, 0.) + np.log(1.+np.exp(-abs(logits_q))) + np.maximum(logits_p, 0.) + np.log(1.+np.exp(-abs(logits_p)))
        closest_dist_sq = KL_mtx(X[:, 0], centers[0]).transpose()
    elif mode == 'ce':
        closest_dist_sq = CE_mtx(X[:, 0], centers[0]).transpose()
    elif mode == 'jsd':
        closest_dist_sq = JSD_mtx(X[:, 0], centers[0]).transpose()
    else:
        raise ValueError("Unknown distance in Kmeans++ initialization")

    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rnd_samples = random_state.random_sample(n_local_trials)
        test1 = random_state.random_sample(n_local_trials)
        rand_vals = rnd_samples * current_pot
        assert np.any(abs(test1 - rnd_samples) > 1e-4)

        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        if mode == 'euclidean':
            distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        elif mode == 'ce':
            distance_to_candidates = CE_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        elif mode == 'kl':
            distance_to_candidates = KL_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        else:
            distance_to_candidates = JSD_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        center_ids[c] = best_candidate
        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers, center_ids

def deepdish_write(fpath, data):
    ''' Save a dictionary as a hdf5 file! '''
    deepdish.io.save(fpath, data, compression="None")

class Logger:
    def __init__(self, fpath):
        self.fpath = fpath
        self.logdata = {}

    def log(self, key, value):
        if key not in self.logdata:
            self.logdata[key] = []
        self.logdata[key].append(value)

    def last(self, key):
        return self.logdata[key][-1]

    def log_dict(self, dictionary, suffix=""):
        # logging each element in the dictionary
        suffix = "_%s" % (suffix) if (suffix != "" and suffix[0] != "_") else suffix
        for k, v in dictionary.items():
            self.log(k+suffix, v)

    def end_log(self):
        for k, v in self.logdata.items():
            self.logdata[k] = np.array(v) if isinstance(v, list) else v

    def save_log(self):
        deepdish_write(self.fpath, self.logdata)

def run_imax(logodds, y, num_bins=15, p_y_pos=None, num_steps=200, init_mode="kmeans", bin_repr_during_optim="pred_prob_based", log_every_steps=10, logfpath=None, skip_slow_evals=True):
    print("Starting I-Max MI Optimization!")
    if p_y_pos is None:
        p_y_pos = y.mean()
    p_y_neg = 1 - p_y_pos

    # fit KDE for MI estimation
    distr_kde_dict = fit_kde_distributions(logodds, y)

    bin_repr_func = bin_representation_function(logodds, y, num_bins, bin_repr_scheme=bin_repr_during_optim)  # get the sample_based or pred_prob_based representations used during training
    bin_boundary_func = bin_boundary_function()
    loss_func = mutual_information_and_R_function(p_y_pos, distr_kde_dict)
    loss_func_unknown = unknown_LLR_mutual_information(p_y_pos, logodds)

    if skip_slow_evals == False:
        # get upper bounds
        H_y, MI_lambda_y = MI_upper_bounds(p_y_pos, distr_kde_dict)
        print("Upper bounds: H_y: %.7f and MI(lambda, y): %.7f" % (H_y, MI_lambda_y))
    else:
        H_y, MI_lambda_y = -1, -1

    if init_mode == "kmeans":
        representations, _ = kmeans_pp_init(logodds[..., np.newaxis], num_bins, 755619, mode='jsd')
        representations = np.sort(np.squeeze(representations))
    elif init_mode == "eqmass" or init_mode == "eqsize" or "custom_range" in init_mode:
        boundaries = nolearn_bin_boundaries(num_bins, binning_scheme=init_mode, x=logodds)
        representations = bin_repr_func(boundaries)
    else:
        raise Exception("I-Max init unknown!")

    LOG = Logger(fpath=logfpath)
    pbar = tqdm(range(num_steps))
    MI, Rbitrate, MIunknownLLR = np.inf, np.inf, np.inf
    for i, step in enumerate(pbar):
        # Boundary update
        boundaries = bin_boundary_func(representations)

        # Theta - bin repr update
        representations = bin_repr_func(boundaries)

        # logging
        if log_every_steps is not None and (step % log_every_steps == 0 or step == 0):
            # Loss calc.
            MI, Rbitrate = loss_func(boundaries)
            MIunknownLLR = loss_func_unknown(boundaries, representations)
            LOG.log("step", step)
            LOG.log("bin_boundaries", boundaries)
            LOG.log("bin_representations", representations)
            LOG.log("MIunknownLLR", MIunknownLLR)
            LOG.log("MI", MI)
            LOG.log("Rbitrate", Rbitrate)

        print_str = "%d/%d, (MI, R) : (%.7f, %.3f) - MIunknownLLR: %.7f" % (step, num_steps, MI, Rbitrate, MIunknownLLR)
        pbar.set_description(("%s" % (print_str)))
    # end all learning steps!!!!
    del pbar
    MI, Rbitrate = loss_func(boundaries)
    MIunknownLLR = loss_func_unknown(boundaries, representations)
    print("(MI, R) = (%.7f, %.3f) - MIunknownLLR: %.7f" % (MI, Rbitrate, MIunknownLLR))
    LOG.log("step", step)
    LOG.log("bin_boundaries", boundaries)
    LOG.log("bin_representations", representations)
    LOG.log("MIunknownLLR", MIunknownLLR)
    LOG.log("MI", MI)
    LOG.log("Rbitrate", Rbitrate)

    print(print_str)
    print("\n")

    LOG.log("H_y", H_y)
    LOG.log("MI_lambda_y", MI_lambda_y)
    LOG.end_log()
    if LOG.fpath is not None:
        LOG.save_log()
    return LOG.logdata

def nolearn_bin_boundaries(num_bins, binning_scheme, x=None):
    """
    Get the bin boundaries (in logit space) of the <num_bins> bins. This function returns only the bin boundaries which do not include any type of learning.
    For example: equal mass bins, equal size bins or overlap bins.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binning_scheme: string
        The way the bins should be placed.
            'eqmass': each bin has the same portion of samples assigned to it. Requires that `x is not None`.
            'eqsize': equal spaced bins in `probability` space. Will get equal spaced bins in range [0,1] and then convert to logodds.
            'custom_range[min_lambda,max_lambda]': equal spaced bins in `logit` space given some custom range.
    x: numpy array (1D,)
        array with the 1D data to determine the eqmass bins.

    Returns
    -------
    bins: numpy array (num_bins-1,)
        Returns the bin boundaries. It will return num_bins-1 bin boundaries in logit space. Open ended range on both sides.
    """
    if binning_scheme == "eqmass":
        assert x is not None and len(x.shape) == 1
        bins = np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1)  # num_bins-1 boundaries for open ended sides
        bins = np.percentile(x, bins * 100, interpolation='lower')  # data will ensure its in Logit space
    elif binning_scheme == "eqsize":  # equal spacing in logit space is not the same in prob space because of sigmoid non-linear transformation
        bins = to_logodds(np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1))  # num_bins-1 boundaries for open ended sides
    elif "custom_range" in binning_scheme:  # used for example when you want bins at overlap regions. then custom range should be [ min p(y=1), max p(y=0)  ]. e.g. custom_range[-5,8]
        custom_range = eval(binning_scheme.replace("custom_range", ""))
        assert type(custom_range) == list and (custom_range[0] <= custom_range[1])
        bins = np.linspace(custom_range[0], custom_range[1], num_bins-1)  # num_bins-1 boundaries for open ended sides
    return bins

class _HistogramBinniner_Binary(BaseCalibrator):
    def __init__(self, cfg, cal_setting, class_idx, num_bins=15, binning_scheme="imax", binning_repr_scheme="pred_prob_based", bin_repr_during_optim="pred_prob_based"):
        """
        Histogram Binning Calibrator. Can be using eqsize, eqmass or Imax binning scheme and different representations.
        This object bins one specific class

        Parameters
        ----------
        cfg: Dict
            all configs from main script.
        class_idx: int or None
            determines which class this obj will bin
        cal_setting: string
            will determine how the data (multi-class) is converted to binary
        binning_scheme: String (default: 'imax')
            scheme to use to determine the binning scheme
        binning_repr_scheme: String (default: pred_prob_based)
            the representation to use when quantizing the logodds during inference
        bin_repr_during_optim: String
            scheme to use to determine the bin representations DURING optimization
        """
        self.cfg = cfg
        self.cal_setting = cal_setting
        self.class_idx = class_idx
        self.num_bins = num_bins
        self.binning_scheme = binning_scheme
        self.binning_repr_scheme = binning_repr_scheme
        self.bin_repr_during_optim = bin_repr_during_optim

        # checks
        assert self.binning_scheme in ["imax", "eqmass", "eqsize"] or "custom_range" in self.binning_scheme
        assert self.bin_repr_during_optim in ["pred_prob_based", "sample_based"]
        assert self.binning_repr_scheme in ["pred_prob_based", "sample_based"]

        self.bin_boundaries = np.zeros(num_bins-1)
        self.bin_representations_SB = np.zeros(num_bins)
        self.bin_representations_PPB = np.zeros(num_bins)
        self.parameter_list = ["bin_boundaries", "bin_representations_SB", "bin_representations_PPB"]

    def fit(self, logits, logodds, y, logodds_for_bin_reprs=None, **kwargs):
        """
        Fit Histogram Binning calibrator. This function will first learn the bin boundaries (IMAX will learn it the others will be determined).
        After learning the bin boundaries, two types of bin representations will be computed.

        logodds_for_bin_reprs can be used to send scaled_logodds (raw logodds scaled using some scaling method). This will then bin the logodds but use the for bin reprs.
        By default will use the logodds to determine the bin reprs.

        Parameters
        ----------

        Returns
        -------
        """
        logits = None  # dont need it so set it to None to be sure its not used!
        if logodds_for_bin_reprs is None:
            logodds_for_bin_reprs = logodds

        # get the class specific logodds
        logodds, y = binary_convertor(logodds, y, self.cal_setting, self.class_idx)
        logodds_for_bin_reprs, _ = binary_convertor(logodds_for_bin_reprs, None, self.cal_setting, self.class_idx)

        if self.binning_scheme == "imax":
            log = run_imax(logodds, y, self.num_bins, num_steps=200, init_mode=self.cfg["Q_init_mode"], bin_repr_during_optim=self.bin_repr_during_optim, log_every_steps=100)
            self.bin_boundaries = log["bin_boundaries"][-1]
            self.MI = log["MI"]
            self.Rbitrate = log["Rbitrate"]
        elif self.binning_scheme == "eqmass" or self.binning_scheme == "eqsize" or "custom_range" in self.binning_scheme:
            self.bin_boundaries = nolearn_bin_boundaries(self.num_bins, binning_scheme=self.binning_scheme, x=logodds)

            # calc. MI at the end now
            distr_kde_dict = fit_kde_distributions(logodds, y)
            self.MI, self.Rbitrate = MI_known_LLR(self.bin_boundaries, p_y_pos=y.mean(), distr_kde_dict=distr_kde_dict)
        else:
            raise Exception("Binning scheme %s unknown!" % (self.binning_scheme))

        assigned = bin_data(logodds, self.bin_boundaries)  # bin the raw logodds and then use scaled logodds to get the predictions
        self.bin_representations_SB = bin_representation_calculation(None, y, self.num_bins, bin_repr_scheme="sample_based", assigned=assigned)
        self.bin_representations_PPB = bin_representation_calculation(logodds_for_bin_reprs, y, self.num_bins, bin_repr_scheme="pred_prob_based", assigned=assigned)

    def calibrate(self, logits, logodds, **kwargs):
        """
        Calibrate using HB. Will also return the bin assignements which are used to calculate the ECE.
        Will always take in multi-class data but only returns the calibrated preds for a binary case which depends on cal_setting and class_idx.

        Parameters
        ----------
        logits: numpy ndarray
           Logits which need to be binned. They will be converted to logodds here.
        Returns
        -------
        cal_logits: numpy ndarray
            calibrated logits (in this case None as bin logodds and not logits)
        cal_logodds: numpy ndarray
            calibrated logodds
        cal_probs: numpy ndarray
            calibrated probabilities
        assigned: numpy array
            bin id assignements. needed to calculate the ECE
        """
        logits = None  # dont need it so set it to None to be sure its not used!

        assert self.cal_setting != "sCW"
        logodds, _ = binary_convertor(logodds, None, self.cal_setting, self.class_idx)

        assigned = bin_data(logodds, self.bin_boundaries)
        if self.binning_repr_scheme == "pred_prob_based":
            bin_reprs = self.bin_representations_PPB
        elif self.binning_repr_scheme == "sample_based":
            bin_reprs = self.bin_representations_SB
        cal_logodds = bin_reprs[assigned]  # fill up representations based on assignments
        cal_probs = to_sigmoid(cal_logodds)  # prob space
        cal_logits = None
        return cal_logits, cal_logodds, cal_probs, assigned

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)

    def save_params(self, fpath):
        raise Exception("Save all binary parameters as one instead of single files")

    def load_params(self, fpath):
        raise Exception("Load all binary parameters from one file instead of single files")
    
def probs_to_logodds(x):
    """
    Use probabilities to convert to logodds. Faster than logits_to_logodds.
    """
    assert x.max() <= 1 and x.min() >= 0
    logodds = to_logodds(x)
    assert np.all(np.isfinite(logodds))
    return logodds

def logits_to_logodds(x):
    """
    Convert network logits directly to logodds (without conversion to probabilities and then back to logodds) using:

    .. math::
        \\lambda_k=z_k-\\log\\sum\\nolimits_{k'\\not = k}e^{z_{k'}}

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array

    axis: int
        Dimension with classes

    Returns
    -------
    logodds : numpy ndarray
       Logodds output
    """
    n_classes = x.shape[1]
    all_logodds = []
    for class_id in range(n_classes):
        logodds_c = x[..., class_id][..., np.newaxis] - custom_logsumexp(np.delete(x, class_id, axis=-1), axis=-1)
        all_logodds.append(logodds_c.reshape(-1))
    logodds = np.stack(all_logodds, axis=1)
    assert np.all(np.isfinite(logodds))
    return logodds

def custom_logsumexp(x, axis=-1):
    """
    Uses the log-sum-exp trick.

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array

    axis: int (default -1)
        axis along which to take the sum

    Returns
    -------
    out: numpy ndarray
        log-sum-exp of x along some axis
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    x_max[~np.isfinite(x_max)] = 0
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    s[s <= 0] = np.finfo(float).eps  # only add epsilon when argument is zero
    out = np.log(s)
    out += x_max
    return out

def to_softmax(x, axis=-1):
    """
    Stable softmax in numpy. Will be applied across last dimension by default.
    Takes care of numerical instabilities like division by zero or log(0).

    Parameters
    ----------
    x : numpy ndarray
       Logits of the network as numpy array.
    axis: int
       Dimension along which to apply the operation (default: last one)

    Returns
    -------
    softmax: numpy ndarray
       Softmax output
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    assert np.all(np.isfinite(softmax)) == True, "Softmax output contains NaNs. Handle this."
    return softmax

def quick_logits_to_logodds(logits, probs=None):
    """
    Using the log-sum-exp trick can be slow to convert from logits to logodds. This function will use the faster prob_to_logodds if n_classes is large.
    """
    n_classes = logits.shape[-1]
    if n_classes <= 100:   # n_classes are reasonable as use this slow way to get marginal
        logodds = logits_to_logodds(logits)
    else:  # imagenet case will always come here!
        if probs is None:
            probs = to_softmax(logits)
        logodds = probs_to_logodds(probs)
    return logodds

@META_ARCH_REGISTRY.register()
class I_Max_sCW(nn.Module):
    def __init__(self,cfg0) -> None:
        super().__init__()
        self.need_calibration_train = True
        self.require_iterative_training = False
        
        cfg = dict(
                # All
                cal_setting="sCW",  # CW, sCW or top1
                num_bins=15,
                # Binning
                Q_method="imax",
                Q_binning_stage="raw",  # bin the raw logodds or the 'scaled' logodds
                Q_binning_repr_scheme="sample_based",
                Q_bin_repr_during_optim="pred_prob_based",
                Q_rnd_seed=928163,
                Q_init_mode="kmeans",
                n_classes=cfg0.MODEL.NUM_CLASS
            )

        self.cfg = cfg
        self.histbin_c = _HistogramBinniner_Binary(cfg, cfg["cal_setting"], None, cfg["num_bins"], cfg["Q_method"], cfg["Q_binning_repr_scheme"], cfg["Q_bin_repr_during_optim"])

    def forward(self,logits,labels):
        all_possible_classes = list(range(self.cfg["n_classes"]))
        encoder = OneHotEncoder(sparse=False,categories=[all_possible_classes])  # sparse=False意味着输出一个numpy数组，而不是稀疏矩阵
        # 将标签数据转换为one-hot编码
        one_hot_labels = encoder.fit_transform(labels.reshape(-1, 1))  # reshape是必要的，因为OneHotEncoder期望2D输入

        if self.training:
            valid_probs = to_softmax(logits)
            valid_logodds = quick_logits_to_logodds(logits, probs=valid_probs)
            self.histbin_c.fit(None, valid_logodds, one_hot_labels, logodds_for_bin_reprs=None)
        else:
            test_probs = to_softmax(logits)
            test_logodds = quick_logits_to_logodds(logits, probs=test_probs)
            assert len(test_logodds.shape) > 1, "Need to send all logodds. Splitting into individual classes will be done in binary scalers."
            cal_logodds = np.zeros_like(test_logodds)
            cal_probs = np.zeros_like(test_logodds)
            cal_assigned = np.zeros_like(test_logodds, dtype=np.int64)
            for class_idx in range(self.cfg["n_classes"]):

                # create a temp calib obj which will calibrate each class using the same binning parameters!
                temp_histbin_c = _HistogramBinniner_Binary(self.cfg, cal_setting="CW", class_idx=class_idx,
                                                        num_bins=self.histbin_c.num_bins, binning_scheme=self.histbin_c.binning_scheme,
                                                        binning_repr_scheme=self.histbin_c.binning_repr_scheme, bin_repr_during_optim=self.histbin_c.bin_repr_during_optim)
                for k in temp_histbin_c.parameter_list:
                    setattr(temp_histbin_c, k, getattr(self.histbin_c, k))  # update each parameters of temp binner with sCW learned version
                _, new_logodds, new_probs, new_assigned = temp_histbin_c(None, test_logodds)
                cal_logodds[..., class_idx] = new_logodds
                cal_probs[..., class_idx] = new_probs
                cal_assigned[..., class_idx] = new_assigned
            cal_probs = torch.from_numpy(cal_probs)
            confidences, predictions = torch.max(cal_probs,dim=1)
            labels = torch.from_numpy(labels)
            return confidences, predictions, labels


