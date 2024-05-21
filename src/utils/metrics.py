from sklearn import metrics
import torch
import numpy as np
#Metrics from https://github.com/runame/laplace-refinement/blob/main/utils/metrics.py

def nll(y_pred, y_true):
    """
    Mean Categorical negative log-likelihood. `y_pred` is a probability vector.
    """
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return metrics.log_loss(y_true, y_pred)

def brier(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        def one_hot(targets, nb_classes):
            res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        return metrics.mean_squared_error(y_pred, one_hot(y_true, y_pred.shape[-1]))

def get_auroc(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return metrics.roc_auc_score(labels, examples)


def get_fpr95(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc) / len(conf_out)
    return fpr.item(), perc.item()

def get_calib(pys, y_true, M=10):
    # Put the confidence into M bins
    pys, y_true = pys.cpu().numpy(), y_true.cpu().numpy()
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    # In percent
    ECE, MCE = ECE*100, MCE*100

    return accs_bin, confs_bin, ECE.item(), MCE.item()

def calculate_ECE_per_bin(y_preds, y_targets, n_bins=10, ECE_type="1"):
    #https://arxiv.org/pdf/1706.04599.pdf

    y_preds, y_targets = y_preds.cpu(), y_targets.cpu()
    
    if ECE_type=="K":
        output_dims = y_preds.shape[1]

        #We flatten probability list which previously had shape (n, K) where n=#test_samples, K=#classes
        y_preds = torch.flatten(y_preds)

        #We create a list of labels with shape (n*K) using repeat interleave. That gives us a label for each
        #p_{n,i}. If previous labels were [0, 1, 0] and K=2, then new list is [0, 0, 1, 1, 0, 0]
        labels = torch.repeat_interleave(y_targets, output_dims)

        #Now we simply need to make a list for comparing to the labels, i.e. [0, 1, 0, 1, .... , 0, 1]
        #until we have (n*K) length list
        probability_labels = torch.tensor(np.arange(output_dims))
        probability_labels = probability_labels.repeat(len(y_targets))
    elif ECE_type=="1":
        prob_list, probability_labels = torch.max(y_preds, dim=1)
        labels = y_targets

    
    #We create the bin limits.
    bin_limits = torch.arange(0,1, 1/n_bins)
    
    ECE_dict = {}
    ECE_dict_count = {}
    weighted_ECE = 0
    
    #We now iterate over the number of bins in ECE calc.
    for i in range(n_bins):
        #If statement to get the correct "bin limits" (i.e. in first bin case limits of bin are (0.0, 0.1)
        #if #bins=10).
        if i != n_bins-1:
            #Here we get all $p_{i, n}$ that lie within this specific bin limits, i.e. first bin
            #we get all indeces of $p_{i, n}$ where probability is in range [0, 0.1) if #bins=10.
            bin_indeces = torch.where((prob_list >= bin_limits[i])&(prob_list < bin_limits[i+1]), True, False)
            bin_mid = (bin_limits[i]+bin_limits[i+1])/2
        else:
            #Same idea as previous if statement, just in the final bin case.
            bin_indeces = torch.where(prob_list >= bin_limits[i], True, False)
            bin_mid = (bin_limits[i]+1)/2
        #Get all prbabilites that "go" in this bin.
        bin_probs = prob_list[bin_indeces]
        #Mean predicted probabilities (confidence) in bin.
        mean_prob = torch.mean(bin_probs)
        
        #Get predictions and labels of $p_{i, n}$ that belong to this bin.
        bin_preds = probability_labels[bin_indeces]
        bin_labels = labels[bin_indeces]
        #Check to ensure that there actaully are samples in this bin, otherwise we get nan.
        if len(bin_probs) != 0:
            #Compute accuracy inside this specific bin.
            accuracy = torch.sum(bin_preds==bin_labels)/len(bin_probs)
            
            #Save accuracy within specific bin in a dictionary. 
            ECE_dict[str(round(bin_mid.item(), 4))] = (accuracy).item()
            ECE_dict_count[str(round(bin_mid.item(), 4))+ " samples in bin"] = len(bin_probs)
            
            #Compute ECE by subtracting accuracy in bin with confidence in bin, and weight it according
            #to number of $p_{i, n}$ that are in this bin out of (n*K).
            weighted_ECE += torch.abs(accuracy-mean_prob).item()*len(bin_probs)/len(prob_list)
        else:
            ECE_dict[str(round(bin_mid.item(), 4))] = 0
    return weighted_ECE, ECE_dict
