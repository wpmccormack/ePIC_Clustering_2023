import numpy as np
import scipy as sp
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy import sparse as sps
from math import log

def weighted_v_score(labels_true, labels_pred, beta=1.0, labels_weight=None):

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C_hat = weighted_entropy(labels_true, weights=labels_weight)
    entropy_K_hat = weighted_entropy(labels_pred, weights=labels_weight)

    contingency_hat = weighted_contingency_matrix(labels_true, labels_pred, sparse=True, weights=labels_weight)
    MI_hat = mutual_info_score(None, None, contingency=contingency_hat)

    homogeneity_hat = MI_hat / (entropy_C_hat) if entropy_C_hat else 1.0
    completeness_hat = MI_hat / (entropy_K_hat) if entropy_K_hat else 1.0

    if homogeneity_hat + completeness_hat == 0.0:
        v_measure_score_hat = 0.0
    else:
        v_measure_score_hat = (
            (1 + beta)
            * homogeneity_hat
            * completeness_hat
            / (beta * homogeneity_hat + completeness_hat)
        )

    return homogeneity_hat, completeness_hat, v_measure_score_hat

def weighted_entropy(labels, weights=None):
    """Calculates the entropy for a labeling."""
    if weights is None:
        weights = np.ones(len(labels))
    
    _, labels = np.unique(labels, return_inverse=True)

    pi_hat = np.bincount(labels, weights=weights)
    pi_hat = pi_hat[pi_hat > 0]
    pi_hat_sum = np.sum(pi_hat)

    return -np.sum((pi_hat / pi_hat_sum) * (np.log(pi_hat) - np.log(pi_hat_sum)))

def weighted_contingency_matrix(labels_true, labels_pred, sparse=False, weights=None):
    """Build a contengency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    sparse : boolean, default False
        If True, return a sparse CSR continency matrix. If 'auto', the sparse
        matrix is returned for a dense input and vice-versa.
    weights : array, shape = [n_samples], optional
        Sample weights.
    """

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    if weights is None:
        weights = np.ones(len(labels_true))

    # Make a float sparse array
    contingency = sp.sparse.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=np.float64
    )

    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        return contingency

    return contingency.toarray() 

def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    
    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        # For a sparse matrix
        nzx, nzy, nz_val = sps.find(contingency)

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    return np.clip(mi.sum(), 0.0, None)
