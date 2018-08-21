import text_data
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
import score_data
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.pipeline import Pipeline
import os

#Tuning Hyperparameters

data=text_data.load_data(type="str")
method="aggl"

# 为了防止全零行计算cosine affinity 出错补上一个标志
def mark_allzeros(matrix):
    mark=np.prod(matrix==0,axis=1)
    return np.concatenate((matrix,mark.reshape((len(mark),1))),axis=1)

if method=="KMeans":
    params={'vect__min_df':[10,15,20],
            'svd__n_components':[100,150,200],
            'KMeans__n_clusters':[17,15,13,11,10]}


    texf_cluster=Pipeline([('vect',text.TfidfVectorizer()),
                           ('svd',TruncatedSVD()),
                           ('KMeans',KMeans(n_init=100,max_iter=2000,random_state=42,n_jobs=1))])

    '''
    gs_cluster=GridSearchCV(estimator=texf_cluster,
                            param_grid=parms,
                            scoring="v_measure_score",
                            cv=[(range(0,len(data)), range(0,len(data)))]) # do not need CV

    parms_result=gs_cluster.fit(data,text_data.labels_true())
    print(parms_result.best_score_)
    print(parms_result.best_params_)
    '''

    result=[]

    for g in list(model_selection.ParameterGrid(params)):
        print()
        print(g)
        texf_cluster.set_params(**g)
        labels_pred=texf_cluster.fit_predict(data)
        print(labels_pred)
        count_table=score_data.count_table(text_data.init_num_by_cls,labels_pred,g['KMeans__n_clusters'])
        print(count_table)
        #total_entropy=score_data.total_entropy(count_table)
        #print("Total Entropy:",total_entropy)
        print("homogeneity score, completeness score, v score:",
              metrics.homogeneity_completeness_v_measure(text_data.labels_true(),labels_pred))
        print("Adjusted Mutual Information:",
              metrics.adjusted_mutual_info_score(text_data.labels_true(),labels_pred))
        result.append((g,
                       metrics.adjusted_mutual_info_score(text_data.labels_true(),labels_pred),
                       metrics.homogeneity_completeness_v_measure(text_data.labels_true(),labels_pred)))

elif method=="aggl":
    params={'vect__min_df':np.arange(5,12,2),
            'svd__n_components':np.arange(80,125,10),
            'n_clusters':np.arange(12,15)}
    texf_svd=Pipeline([('vect',text.TfidfVectorizer()),
                           ('svd',TruncatedSVD())])
    result=[]
    Aggl=AgglomerativeClustering(compute_full_tree=True,affinity='cosine',linkage='average')
    for g in list(model_selection.ParameterGrid(params)):
        print()
        print(g)
        n_clusters=g.pop('n_clusters')
        texf_svd.set_params(**g)
        svd_result=texf_svd.fit_transform(data)
        Aggl.set_params(n_clusters=n_clusters,memory=os.getcwd()+"\\tree")
        labels_pred=Aggl.fit_predict(mark_allzeros(svd_result))
        print(labels_pred)
        count_table=score_data.count_table(text_data.init_num_by_cls,labels_pred,n_clusters)
        print("homogeneity score, completeness score, v score:",
              metrics.homogeneity_completeness_v_measure(text_data.labels_true(),labels_pred))
        print("Adjusted Mutual Information:",
              metrics.adjusted_mutual_info_score(text_data.labels_true(),labels_pred))
        result.append((g,
                       metrics.adjusted_mutual_info_score(text_data.labels_true(),labels_pred),
                       metrics.homogeneity_completeness_v_measure(text_data.labels_true(),labels_pred)))

