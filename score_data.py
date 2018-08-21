import numpy as np
import collections
from scipy import stats

def count_table(classes,cluster_labels,n_clusters):
    n_classes=len(classes)
    table=np.zeros((n_classes,n_clusters),dtype=int)
    last=0
    for i in range(n_classes):
        class_count=collections.Counter(cluster_labels[last:last+classes[i]])
        last+=classes[i]
        table[i]=np.array([class_count[j] for j in range(n_clusters)])
    return table


def total_entropy(count_table):
    '''equivalent to homogeneity score'''
    n_j=np.sum(count_table,axis=0)
    entropy_j=np.array([stats.entropy(count_table[:,j]) for j in range(count_table.shape[1])])
    return -sum(n_j*entropy_j)/sum(n_j)