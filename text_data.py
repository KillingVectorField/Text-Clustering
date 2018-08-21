import os
from tqdm import tqdm
import json
import numpy as np
import collections

class_name=["财经","房产","健康","教育","军事","科技","体育","娱乐","证券"]
folderpath=[os.getcwd()+"\\new_weibo_13638\\"+x for x in class_name]
init_num_by_cls=np.array([2375, 1211, 670, 447, 791, 1397, 3325, 2255, 1167])

def is_num(s):
    '''判断str是不是numeric'''
    ret = False
    try:
        _ = float(s)
        ret = True
    except ValueError:
        pass
    return ret

#遍历指定文件夹，显示目录下的所有文件名
def eachFile(folderpath):
    pathDir =  os.listdir(folderpath)
    filepath=[os.path.join(folderpath,allDir) for allDir in pathDir]
    return filepath
        
# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r', encoding='utf8',errors='ignore') # r 代表read
    str=fopen.read()
    fopen.close()
    return str

def str2type(str,type="dict"):#将文本转化为集合（不考虑同一个词重复出现的次数）或字典
    word_list=str[:-1].split(sep='\t') # 最后一个字符是 "\n"
    for word in word_list:
        if is_num(word):    #数字不管
            word_list.remove(word)
    if type=="set": return set(word_list)
    elif type=="dict": return collections.Counter(word_list)
    elif type=="list": return word_list


def load_data(y_name=class_name,type="dict"):
    data=[]
    for cls in tqdm(class_name):
        allfiles=eachFile (os.getcwd()+"\\new_weibo_13638\\"+cls)
        if type=="str":
            new_class=[readFile(file)[:-1] for file in allfiles]
        else:
            new_class=[str2type(readFile(file),type) for file in allfiles]
        data.extend(new_class)
    return data

def save_to_file(file_name, contents):
    fh = open(file_name, 'w',encoding='utf8')
    fh.write(contents)
    fh.close()

def save_json(tree,file_name='json_file.txt'):
    '''将变量写入json文件'''
    with open(file_name, 'w',encoding='utf8') as file_obj:
        json.dump(tree, file_obj)
        print("已写入json文件",file_name)

def load_json(file_name='json_file.txt') :
    '''从json文件读取变量'''
    with open(file_name) as file_obj:
        loading = json.load(file_obj)  # 返回列表数据，也支持字典
        print("已读取json文件",file_name)
        return loading

def labels_true(num_by_cls=init_num_by_cls):
    labels=[]
    for i,cls in enumerate(num_by_cls):
        labels+=[i]*cls
    return np.array(labels)