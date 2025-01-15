import pickle

# Open file with pickled variables
def unpickle_g_logits(file, verbose = 0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        ((train_features,train_logits,train_labels),(valid_features, valid_logits, valid_labels), (test_features, test_logits, test_labels)) = pickle.load(f)  # unpickle the content
        
    if verbose:    
        print("y_probs_val:", valid_logits.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", valid_labels.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", test_logits.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", test_labels.shape)  # (10000, 1); Test set true labels
        
    return ((train_features,train_logits,train_labels),(valid_features, valid_logits, valid_labels), (test_features, test_logits, test_labels))