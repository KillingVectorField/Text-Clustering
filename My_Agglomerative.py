import time

import numpy as np
from scipy import cluster, spatial
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


def tri2array(n,i,j):
    '''
    从上三角矩阵的index(i,j)转成condensed array的下标
    必须有i<j
    '''
    # assert np.prod(i<j)
    return np.int32(n*(n-1)/2 - (n-i)*(n-i-1)/2 + j - i - 1)

def row_col_from_condensed_index(d,i):
    b = 1 -2*d 
    x = int(np.floor((-b - np.sqrt(b**2 - 8*i))/2))
    y = int(i + x*(b + x + 2)/2 + 1)
    return (x,y)  

def MyAgglomerativeClustering(Dist,linkage="average"):
    '''
    Dist为一维数组（长度为 K(K-1)/2）(scipy.spatial.distance.pdist的返回值）
    n_clusters是长度为K的一维数组，其元素为每个聚类里的元素个数，至少为1
    '''
    n_init=int(np.ceil(np.sqrt(len(Dist)*2)))
    n_clusters=np.ones((n_init,),dtype=np.int32)
    n=len(n_clusters)#聚类个数n
    record=[]
    for i_th in tqdm(range(n-2)):
        #t0=time.clock()
        #triu=np.array(np.triu_indices(n,1))#1对角矩阵，记录Dist下标到i,j的映射，i<j
        max_indice=Dist.argmin()
        #t1=time.clock()
        #print('t1:',t1-t0)
        i,j=row_col_from_condensed_index(n,max_indice)
        #t2=time.clock()
        #print('t2:',t2-t1)
        n_i,n_j=n_clusters[i],n_clusters[j]
        n_clusters[i]=n_tot=n_i+n_j
        n_clusters=np.delete(n_clusters,j)
        record.append([i,j,Dist[max_indice],n_tot])
        Dist_old=np.copy(Dist)
        #t2_5=time.clock()
        #print('t2_5:',t2_5-t2)
        #删除第j各元素相关的元素，共(n-1)个
        Dist=np.delete(Dist,tri2array(n,j,np.arange(n-1,j,-1)))
        Dist=np.delete(Dist,tri2array(n,np.arange(j-1,-1,-1),j))
        #t3=time.clock()
        #print('t3:',t3-t2_5)
        # 此时Dist已经形如(n-1)个元素的condensed distance array
        # 更新Dist中与i有关的元素，共(n-2)个
        # D(k,ij)=(n_i * D(k,i)+n_j*D(k,j))/(n_i+n_j)
        k=np.arange(i)
        Dist[tri2array(n-1,k,i)]=(n_i*Dist_old[tri2array(n,k,i)]+
                                  n_j*Dist_old[tri2array(n,k,j)])/n_tot
        k=np.arange(i+1,j)
        Dist[tri2array(n-1,i,k)]=(n_i*Dist_old[tri2array(n,i,k)]+
                                  n_j*Dist_old[tri2array(n,k,j)])/n_tot
        k=np.arange(j+1,n)
        Dist[tri2array(n-1,i,k-1)]=(n_i*Dist_old[tri2array(n,i,k)]+
                                  n_j*Dist_old[tri2array(n,j,k)])/n_tot
        #t4=time.clock()
        #print('t4:',t4-t3)
        n=len(n_clusters)
    # n=2 时
    record.append([0,1,Dist[0],n_clusters[0]+n_clusters[1]])
    return record

def MyAggl_pred(record,n_clusters,re="predict"):
    '''
    从MyAggl函数的返回值中推断聚类结果
    record的shape为(n-1,2)
    '''
    n=record.shape[0]+1
    clusters=[[i] for i in range(n)]
    for i_th in range(n-n_clusters):
        clusters[record[i_th,0]].extend(clusters[record[i_th,1]])
        del clusters[record[i_th,1]]
    if re=="clusters": return clusters
    else:
        pred=np.zeros(n,dtype=int)
        for i,clt in enumerate(clusters):
            pred[clt]=np.array([i]*len(clt))
        if re=="predict": return pred
        elif re=="both": return (y,pred)
    raise ValueError("re can only be ['clusters','predict','both']")

#y=np.arange(16).reshape(4,4)
y=np.random.randn(100,50)
#y=np.random.randn(13000,100)
start=time.clock()
Dist=spatial.distance.pdist(y,metric="cosine")
pdist_time=time.clock()
print("pdist:",pdist_time-start)
sci_result=cluster.hierarchy.linkage(Dist,method="average")
sci_time=time.clock()
print("scipy:",sci_time-pdist_time)
my_result=MyAgglomerativeClustering(Dist)
myaggl_time=time.clock()
print("My Agglometative:",myaggl_time-sci_time)
#print(sci_result)
#print(my_result)
my_pred=MyAggl_pred(np.array(my_result,dtype=int)[:,:2],n_clusters=10)
#print(my_pred)


#验证与sklearn 库的结果一致
sk_Aggl=AgglomerativeClustering(n_clusters=10,affinity="cosine",linkage="average")
sci_pred=sk_Aggl.fit_predict(y)
print(sci_pred)
print("Adjusted Mutual Information:",
                metrics.adjusted_mutual_info_score(my_pred,sci_pred))
