import text_data
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
import score_data
import numpy as np
from sklearn import metrics
from scipy import spatial,cluster

use_idf=True
least_freq=5
n_clusters=14
n_components=100
#KMeans_n_init=100
#KMeans_max_iter=1000
affinity='cosine'
linkage='average'


data=text_data.load_data(type="str")

countVectorizer=text.TfidfVectorizer(input='content',min_df=least_freq, use_idf=use_idf)# 最少总共出现过min_df次
term_freq=countVectorizer.fit_transform(data)
print('词典总词数:',len(countVectorizer.vocabulary_))
print(countVectorizer.vocabulary_)


svd=TruncatedSVD(n_components=n_components)
svd_result=svd.fit_transform(term_freq)
print("Explained variance of the SVD step: {}%".format(
    int(svd.explained_variance_ratio_.sum() * 100)))

# 为了防止全零行计算cosine affinity 出错补上一个标志
def mark_allzeros(matrix):
    mark=np.prod(matrix==0,axis=1)
    return np.concatenate((matrix,mark.reshape((len(mark),1))),axis=1)


'''
Sci_KMeans=KMeans(n_clusters=n_clusters,random_state=42,
                    n_init=KMeans_n_init,max_iter=KMeans_max_iter)
labels_pred=Sci_KMeans.fit_predict(svd_result)
'''

marked_svd=mark_allzeros(svd_result)

#sk_Aggl=AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,linkage=linkage)
#sk_pred=sk_Aggl.fit_predict(marked_svd)

Dist=spatial.distance.pdist(marked_svd,metric="cosine")
sci_result=cluster.hierarchy.linkage(Dist,method="average")
from matplotlib import pyplot as plt
dn=cluster.hierarchy.dendrogram(sci_result)
plt.show()

my_pred=myag.MyAgglomerativeClustering(Dist)
label_pred=myag.MyAggl_pred(np.array(my_pred,dtype=int)[:,:2],n_clusters=n_clusters)
text_data.save_json([int(i) for i in label_pred],filename)

label_pred=text_data.load_json(filename)

for labels_pred in [my_pred]:
    count_table=score_data.count_table(text_data.init_num_by_cls,labels_pred,n_clusters)
    print(count_table)
    total_entropy=score_data.total_entropy(count_table)
    print("Total Entropy:",total_entropy)
    print("homogeneity_score",metrics.homogeneity_completeness_v_measure(text_data.labels_true(),labels_pred))

