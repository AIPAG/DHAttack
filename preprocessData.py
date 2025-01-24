import sys
from tkinter import Y
from types import new_class
sys.dont_write_bytecode = True   

import numpy as np
import math


import torch.nn.functional as F

import random
import os

import readData as rd  
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR



from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from scipy.interpolate import interp1d

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse

np.random.seed(21312)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_data_for_trainortest(data_name):   
    with np.load( data_name) as f:
        
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]  
    return train_x, train_y

def load_data_for_attack(data_name):   
    with np.load( data_name) as f:
        
        train_x, train_y, data_label = [f['arr_%d' % i] for i in range(len(f.files))]  
    return train_x, train_y, data_label





def getIndexByValue(dataYList,label):
    indexList = []
    for index, value in enumerate(dataYList):
        if value == label:
            indexList.append(index)
    return indexList



def shuffleAndSplitDataByClass(dataX, dataY,cluster):  
       
    
    n_class = np.unique(dataY) 
    cluster_for_one_class = math.floor(cluster/len(n_class)) 
    toTrainDataIndexTotal = []
    shadowDataIndexTotal = []
    toTestDataIndexTotal = []
    shadowTestDataIndexTotal = []
    distillationDataIndexTotal = []
    dataYList = dataY.tolist()
    
    print("total number of samples: {}, total number of classes:{}, number of samples per class:{}".format(len(dataYList), len(n_class), cluster_for_one_class))
    print("number of distillation samples is {}".format(len(dataYList)-4*cluster_for_one_class*len(n_class)))
    for label in n_class:		
        cluster_for_one_class = math.floor(cluster/len(n_class))
        dataIndex = getIndexByValue(dataYList, label) 
        

        random.shuffle(dataIndex)   
        
        if math.floor(len(dataIndex)/5) < cluster_for_one_class: 
            cluster_for_one_class = math.floor(len(dataIndex)/5)
        
        toTrainDataIndex  = np.array(dataIndex[:cluster_for_one_class])
        shadowDataIndex  = np.array(dataIndex[cluster_for_one_class:cluster_for_one_class*2])
        toTestDataIndex  = np.array(dataIndex[cluster_for_one_class*2:cluster_for_one_class*3])
        shadowTestDataIndex  = np.array(dataIndex[cluster_for_one_class*3:cluster_for_one_class*4])
        distillationDataIndex = np.array(dataIndex[cluster_for_one_class*4:])

        print("class {} has {}, {}, {}, {} samples for targetModel and shadowModel, {} samples for distillation.".format(label, len(toTrainDataIndex), len(shadowDataIndex), len(toTestDataIndex), len(shadowTestDataIndex), len(distillationDataIndex)))

        toTrainDataIndexTotal.append(toTrainDataIndex)
        shadowDataIndexTotal.append(shadowDataIndex)
        toTestDataIndexTotal.append(toTestDataIndex)
        shadowTestDataIndexTotal.append(shadowTestDataIndex)
        distillationDataIndexTotal.append(distillationDataIndex)

    
    toTrainDataIndexTotal = np.concatenate(toTrainDataIndexTotal).astype(np.int64)
    shadowDataIndexTotal = np.concatenate(shadowDataIndexTotal).astype(np.int64)
    toTestDataIndexTotal = np.concatenate(toTestDataIndexTotal).astype(np.int64)
    shadowTestDataIndexTotal = np.concatenate(shadowTestDataIndexTotal).astype(np.int64)
    distillationDataIndexTotal = np.concatenate(distillationDataIndexTotal).astype(np.int64)

    
    random.shuffle(toTrainDataIndexTotal)  
    random.shuffle(shadowDataIndexTotal)
    random.shuffle(toTestDataIndexTotal)
    random.shuffle(shadowTestDataIndexTotal)
    random.shuffle(distillationDataIndexTotal)

    
    toTrainData = np.array(dataX[toTrainDataIndexTotal])  #(15000, 3072)
    toTrainLabel = np.array(dataY[toTrainDataIndexTotal]) #(15000,)
    
    shadowData  = np.array(dataX[shadowDataIndexTotal])
    shadowLabel = np.array(dataY[shadowDataIndexTotal])
    
    toTestData  = np.array(dataX[toTestDataIndexTotal])
    toTestLabel = np.array(dataY[toTestDataIndexTotal])
    
    shadowTestData  = np.array(dataX[shadowTestDataIndexTotal])
    shadowTestLabel = np.array(dataY[shadowTestDataIndexTotal])

    distillationData  = np.array(dataX[distillationDataIndexTotal])
    distillationDataLabel = np.array(dataY[distillationDataIndexTotal])

    return toTrainData, toTrainLabel,   shadowData,shadowLabel,    toTestData,toTestLabel,      shadowTestData,shadowTestLabel,    distillationData, distillationDataLabel
    





def clipDataTopX(dataToClip, top=3):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]   
    return np.array(res)






def initializeDataIncludingDistillationData(dataset,orginialDatasetPath,dataFolderPath = './data/'):  
    
    if(dataset == 'CIFAR10'):    
        print("Loading data (CIFAR10)")
        dataX, dataY, testdataX, testdataY = rd.readCIFAR10(orginialDatasetPath)  
        
        
        X = np.concatenate((dataX , testdataX), axis=0)   #(60000, 3072)
        y = np.concatenate((dataY , testdataY), axis=0) #len(y)  60000
       
        print("{} samples in the dataset".format(len(X)))
        

        cluster = 10000    
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'   

        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y, cluster)  
        
        
        
        toTrainDataSave, toTestDataSave    = rd.preprocessingCIFAR(toTrainData, toTestData)   
        shadowDataSave, shadowTestDataSave = rd.preprocessingCIFAR(shadowData, shadowTestData)   
        distillationDataSave = rd.preprocessingCIFAR(distillationData, np.array([]))     

    elif(dataset == 'News'):
        print("Loading data from internet (News)")
        
        newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')  )    
        newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes') )
        

        
        X = np.concatenate((newsgroups_train.data , newsgroups_test.data), axis=0)   #len(X)  18846
        y = np.concatenate((newsgroups_train.target , newsgroups_test.target), axis=0) #len(y)  18846
        
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        X = X.toarray()
        

        print("Preprocessing data")
        print(X.shape)   #(18846, 134410)
        cluster = 3000   
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'  
        
        
        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y,cluster)
        
        
        toTrainDataSave, toTestDataSave    = rd.preprocessingNews(toTrainData, toTestData)
        shadowDataSave, shadowTestDataSave = rd.preprocessingNews(shadowData, shadowTestData)
        distillationDataSave = rd.preprocessingNews(distillationData, np.array([]))

    elif (dataset == 'CIFAR100'):
        print("Loading data (CIFAR100)")
        dataX, dataY, testX, testY = rd.readCIFAR100(orginialDatasetPath)  
        	

        
        X = np.concatenate((dataX , testX), axis=0)  
        y = np.concatenate((dataY , testY), axis=0) 

        
        print("{} samples in the dataset".format(len(X)))
        

        cluster = 10000  
           
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'  

         
        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y,cluster)  
            
        
        toTrainDataSave, toTestDataSave    = rd.preprocessingCIFAR(toTrainData, toTestData)   
        shadowDataSave, shadowTestDataSave = rd.preprocessingCIFAR(shadowData, shadowTestData)   
        distillationDataSave = rd.preprocessingCIFAR(distillationData, np.array([]))     

    elif (dataset == 'CINIC10'):
        print("Loading data (CINIC10)")
        dataX, dataY, testX, testY, validX, validY = rd.readCINIC10(orginialDatasetPath)  
        
        X = np.concatenate((dataX , testX, validX), axis=0)   #(270000, 3072)
        y = np.concatenate((dataY , testY, validY), axis=0) #len(y)  (270000,)

        
        print("{} samples in the dataset".format(len(X)))
        

        cluster = 10000   #10000*4 = 40000
        
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'   

        
        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y,cluster)  
            
        
        toTrainDataSave, toTestDataSave    = rd.preprocessingCIFAR(toTrainData, toTestData)   
        shadowDataSave, shadowTestDataSave = rd.preprocessingCIFAR(shadowData, shadowTestData)   
        distillationDataSave = rd.preprocessingCIFAR(distillationData, np.array([]))     
    elif (dataset == 'GTSRB'):
        print("Loading data (GTSRB)")
        dataX, dataY, testX, testY = rd.readGTSRB(orginialDatasetPath)  

        
        X = np.concatenate((dataX , testX), axis=0)   
        y = np.concatenate((dataY , testY), axis=0) 

        
        print("{} samples in the dataset".format(len(X)))
          

        cluster = 1500   
        
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'  

        
        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y,cluster)  
        
        toTrainDataSave, toTestDataSave    = rd.preprocessingCIFAR(toTrainData, toTestData)   
        shadowDataSave, shadowTestDataSave = rd.preprocessingCIFAR(shadowData, shadowTestData)   
        distillationDataSave = rd.preprocessingCIFAR(distillationData, np.array([]))     
    
    elif(dataset == 'Purchase100'):
        print("Loading data (Purchase100)")

        X, y = rd.load_data_for_Purchase(orginialDatasetPath)  
        
        print("Preprocessing data")
        print(X.shape)   
        
        cluster = 20000  
        
        dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'  
        
        
        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass(X, y,cluster)		
        toTrainDataSave = toTrainData
        shadowDataSave = shadowData
        toTestDataSave = toTestData
        shadowTestDataSave = shadowTestData
        distillationDataSave = distillationData



    try:
        os.makedirs(dataPath)
    except OSError:
        pass
    
    
    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz',  toTestDataSave, toTestLabel)
    np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
    np.savez(dataPath + '/shadowTest.npz',  shadowTestDataSave, shadowTestLabel)
    np.savez(dataPath + '/distillationData.npz',  distillationDataSave, distillationDataLabel)
    
    print("Preprocessing finished\n\n")


def init_args():
    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    
    dataset = args.dataset  

    if dataset == 'CIFAR10':
        pathToLoadData = './data/cifar-10-download'  
    elif dataset == 'CIFAR100':
        pathToLoadData = './data/cifar-100-download'
    elif dataset == 'CINIC10':
        pathToLoadData = './data/CINIC-10-download'
    elif dataset == 'GTSRB':
        pathToLoadData = './data/GTSRB-download'
    
    else:
        print("Wrong direction! Please see readme.")
    


    
    initializeDataIncludingDistillationData(dataset,pathToLoadData)  


    

        
