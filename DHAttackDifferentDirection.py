import sys
from tkinter import Y
from types import new_class
sys.dont_write_bytecode = True   

import numpy as np
import math
import seaborn as sns


import torch.nn.functional as F

import random
import os
import Models as models  
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

from tqdm import tqdm
from sklearn.metrics import f1_score

import scipy.stats
import time
import argparse

np.random.seed(21312)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_data_for_trainortest(data_name):   
    with np.load( data_name) as f:
        
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]  
    return train_x, train_y




def train_target_model(dataset,epochs=100, batch_size=100, learning_rate=0.001, l2_ratio=1e-7,
                       n_hidden=50, model='nn', datasetFlag='CIFAR10'): 
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_x, train_y, test_x, test_y = dataset
    
    
     
    
    n_out = len(np.unique(train_y))
    

    
    if batch_size > len(train_y):
        batch_size = len(train_y)

    
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

    
    if(datasetFlag=='CIFAR10' or datasetFlag=='CIFAR100' or datasetFlag=='CINIC10' or datasetFlag=='GTSRB'):
        
        train_data = models.CIFARData(train_x, train_y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loader_noShuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)

        
        test_data = models.CIFARData(test_x, test_y)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif(datasetFlag=='News' or datasetFlag=='Purchase100'):
        
        train_data = models.NewsData(train_x, train_y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loader_noShuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)

        
        test_data = models.NewsData(test_x, test_y)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    
    params = {}
    if(datasetFlag=='CIFAR10'):
        params['task'] = 'cifar10'
        params['input_size'] = 32
        params['num_classes'] = 10
    elif(datasetFlag=='CIFAR100'):
        params['task'] = 'cifar100'
        params['input_size'] = 32
        params['num_classes'] = 100
    elif(datasetFlag=='CINIC10'):
        params['task'] = 'cinic10'
        params['input_size'] = 32
        params['num_classes'] = 10
    elif(datasetFlag=='GTSRB'):
        params['task'] = 'gtsrb'
        params['input_size'] = 32
        params['num_classes'] = 43
    
    
    if model == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        net = models.CNNNet(n_hidden=n_hidden, n_out=n_out)
        net = net.to(device)
    elif model == 'nn':
        n_in = train_x.shape[1]
        print('Using a multilayer neural network based model...')
        net = models.NNNet(n_in, n_hidden, n_out) #n_hidden=128
        net = net.to(device)
    elif model =='vgg':
        print('Using vgg model...')
        
        params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        params['fc_layers'] = [512, 512]
        params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        params['conv_batch_norm'] = True
        params['init_weights'] = True   
        params['augment_training'] = True
        net = models.VGG(params)
        net = net.to(device)

    elif model == 'resnet':
        print('Using resnet model...')
        params['block_type'] = 'basic'
        params['num_blocks'] = [9,9,9]    
        params['augment_training'] = True 
        params['init_weights'] = True  
        net = models.ResNet(params)
        net = net.to(device)


    elif model == 'mobilenet':
        print('Using mobilenet model...')
        params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        params['augment_training'] = True
        net = models.MobileNet(params)
        net = net.to(device)

        
    
    net.train()
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    
    if  model == 'vgg' :
        l2_ratio = 0.0005  
        learning_rate = 0.1   

        
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

        
        momentum = 0.9
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_ratio)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9) 

        if datasetFlag == 'CIFAR100':
            epochs = 100  

    
    elif model == 'cnn': 
        epochs = 80
        l2_ratio = 1e-07
        learning_rate = 0.001   
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9) 

    elif model == 'resnet' or model == 'mobilenet':
        l2_ratio = 0.0001
        learning_rate = 0.01   
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9) 
    elif model == 'nn':
        l2_ratio = 1e-7
        learning_rate = 0.01   
        
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
        
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7) 
    
    print('dataset: {},  model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(datasetFlag, model, device, epochs, batch_size, learning_rate, l2_ratio))
    count = 1
    print('Training...')
    for epoch in range(epochs):
        running_loss = 0


        for step, (X_vector, Y_vector) in enumerate(train_loader):	
            

            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)

            output = net(X_vector)
            

             
            loss = criterion(output, Y_vector)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            running_loss += loss.item()

            

            

            
        if optimizer.param_groups[0]['lr']>0.0005:  
            scheduler.step()  

        if (epoch + 1) % 10 == 0: 
            
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))

    print("Training finished!")

    
    pred_y = []
    net.eval()
    if batch_size > len(train_y):
        batch_size = len(train_y)
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):	
        
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output = net(X_vector)
        out_y = output.detach().cpu() 
        pred_y.append(np.argmax(out_y, axis=1))  
        
    pred_y = np.concatenate(pred_y)  
    print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))  
    


    
    print('Testing...')
    pred_y = []
    net.eval()

    
    if batch_size > len(test_y):
        batch_size = len(test_y)
    for step, (X_vector, Y_vector) in enumerate(test_loader):	
        
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output = net(X_vector)
        out_y = output.detach().cpu() 
        pred_y.append(np.argmax(out_y, axis=1))  

        

    


    pred_y = np.concatenate(pred_y)  
    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))  
    print('More detailed results:')
    print(classification_report(test_y, pred_y))   


    
    attack_x, attack_y = [], []
    classification_y = []
    
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output = net(X_vector)
        out_y = output.detach().cpu()
        
        softmax_y = softmax(out_y.numpy())
        

        Y_vector = Y_vector.detach().cpu()
        
        attack_x.append(softmax_y)  
        attack_y.append(np.ones(len(Y_vector)))  
        classification_y.append(Y_vector)
        
        
    
    for step, (X_vector, Y_vector) in enumerate(test_loader):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output = net(X_vector)
        out_y = output.detach().cpu()
        
        softmax_y = softmax(out_y.numpy())
        

        Y_vector = Y_vector.detach().cpu()

        attack_x.append(softmax_y)
        attack_y.append(np.zeros(len(Y_vector)))
        classification_y.append(Y_vector)
        
    
    attack_x = np.vstack(attack_x)   
    attack_y = np.concatenate(attack_y)   
    classification_y = np.concatenate(classification_y) #

    
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')

    return attack_x, attack_y, net, classification_y



def trainTarget(modelType, X, y,
                X_test=[], y_test =[],
                splitData=True,
                test_size=0.5, 
                inepochs=50, batch_size=300,
                learning_rate=0.001, datasetFlag='CIFAR10'):
    
    if(splitData):  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        
        X_train = X
        y_train = y

    
    dataset = (X_train.astype(np.float32),
               y_train.astype(np.int32),
               X_test.astype(np.float32),
               y_test.astype(np.int32))

    
    attack_x, attack_y, theModel, classification_y = train_target_model(dataset=dataset, epochs=inepochs, batch_size=batch_size,learning_rate=learning_rate,
                   n_hidden=128,l2_ratio = 1e-07,model=modelType, datasetFlag=datasetFlag)

    return attack_x, attack_y, theModel, classification_y







def ROC_AUC_Result_logshow(label_values,predict_values,reverse=False):
    if reverse:
        pos_label = 0   
        print('AUC = {}'.format(1-roc_auc_score(label_values,predict_values)))
        final_auc = 1-roc_auc_score(label_values,predict_values)
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values,predict_values)))
        final_auc = roc_auc_score(label_values,predict_values)
    fpr,tpr,thresholds = roc_curve(label_values,predict_values,pos_label=pos_label)  
    
    roc_auc = auc(fpr,tpr)
    plt.title('Receiver Operating Characteristic(ROC)')
    
    plt.loglog(fpr,tpr,'b', label='AUC=%0.4f' %roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0.001,1],[0.001,1],'r--')
    plt.xlim([0.001,1.0])
    plt.ylim([0.001,1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
    
    ax = plt.gca()
    
    line = ax.lines[0]
    
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    f = interp1d(xdata, ydata)
    
    fpr_0 = 0.001

    tpr_0 = f(fpr_0)
    
    print('TPR at 0.001 FPR is {}'.format(tpr_0))

    plt.cla()

    return tpr_0, final_auc



def softmax(x):
   
   shift = np.amax(x, axis=1) 
   shift = shift.reshape(-1,1)  
   x = x - shift    

   exp_values = np.exp(x)
   denominators = np.sum(np.exp(x), axis=1) 
   softmax_values = (exp_values.T/denominators).T  
   return softmax_values




def add_mask(image,level):  
    ch, row, col = image.shape
    mean = 0
    max = np.max(image)
    min = np.min(image)
    mask = max - image  
    mask = level*mask
    noisy = image + mask
    
    return noisy

def add_mask_black(image,level):  
    ch, row, col = image.shape
    mean = 0
    max = np.max(image)
    min = np.min(image)
    mask = image-min  
    mask = level*mask
    noisy = image - mask
    
    return noisy

def add_mask_bydirection(image,level,direction):  
    ch, row, col = image.shape
    
    mask = direction - image  
    mask = level*mask
    noisy = image + mask
    
    return noisy



def computeLDistance(pred_y_noisy, noisy_groundtruth): 
    lenofwindow = int(max(len(pred_y_noisy)/100, 1))  
    zeros = np.zeros(lenofwindow, dtype=int)  
    result = np.equal(pred_y_noisy, noisy_groundtruth).astype(int) 
    try:
        falsePositions = np.where(result==0)[0]  
        firstfalsePosition = falsePositions[0]  
    except IndexError:  
        return len(pred_y_noisy) 

    result = ''.join([str(i) for i in result])
    zeros = ''.join([str(i) for i in zeros])

    if firstfalsePosition==0:
        count = 0
        while result[count]=='0':
            count = count + 1
            if count==len(pred_y_noisy):
                break
        distance = float(lenofwindow) * np.exp(-count)
    else:    
        index = result.find(zeros)  
        if index !=-1:
            distance = index
        else:
            distance = len(pred_y_noisy) 
    
    return distance
    

def compute_tpr_at_LowFPR(labels, predict_values):  
    fpr, tpr, thresholds = roc_curve(labels, predict_values)
    idx = np.where(fpr < 0.001)[0][-1]
    tpr_at_fpr_0_001 = tpr[idx]
    return tpr_at_fpr_0_001

def compute_f1(labels, predict_values):
    f1 = f1_score(labels, predict_values)
    return f1

def compute_AUC(labels, predict_values):
    fpr, tpr, thresholds = roc_curve(labels, predict_values)
    roc_auc = auc(fpr,tpr)
    return roc_auc


def get_file_name(path):  
    model_file_name = []
    final_file_name = []
    files = os.listdir(path)  
    for i in files:
        model_file_name.append(i.replace("distilledModel_","")) 
  
    model_file_name.sort(key=lambda x: int(x[:x.find(".")]))  
    for j in model_file_name:
        final_file_name.append("distilledModel_"+j) 
    
    return final_file_name

def DistillModelUsingHardLabel(original_model,dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './models/',classifierType = 'cnn', TargetOrShadow='Target'):
    
    dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'
    modelPath = modelFolderPath + dataset + '/HardLabel'
    
    
    print("Training the ref_model of {} for {} epoch".format(TargetOrShadow, num_epoch))


    
    distill_x, distill_y  = load_data_for_trainortest(dataPath + '/distillationData.npz')
    
    distill_dataset = (distill_x.astype(np.float32),
               distill_y.astype(np.int32))
    
    
    distilled_model = distill_original_modelHardLabel(modelPath, original_model, classifierType, distill_dataset, epochs=num_epoch, batch_size=100, learning_rate=0.001, l2_ratio=1e-7,
                       n_hidden=128, datasetFlag=dataset, TargetOrShadow=TargetOrShadow)
    
    return distilled_model


def distill_original_modelHardLabel(modelPath, original_model, modelType, distill_dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, datasetFlag='CIFAR10', TargetOrShadow='Target'):
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    distill_x, distill_y = distill_dataset
    
    n_out = len(np.unique(distill_y))
    if TargetOrShadow =='Target':
        print('(Hard Label) Relabel target_model with {} samples, {} classes...'.format(len(distill_x), n_out))
    elif TargetOrShadow =='Shadow':
        print('(Hard Label) Relabel shadow_model with {} samples, {} classes...'.format(len(distill_x), n_out))
    
   
    if(datasetFlag=='CIFAR10' or datasetFlag=='CIFAR100' or datasetFlag=='CINIC10' or datasetFlag=='GTSRB'):
        
        distill_data = models.CIFARData(distill_x, distill_y)
        distill_loader_noShuffle = DataLoader(distill_data, batch_size=batch_size, shuffle=False)

    elif(datasetFlag=='News' or datasetFlag=='Purchase100'):
        
        distill_data = models.NewsData(distill_x, distill_y)
        distill_loader_noShuffle = DataLoader(distill_data, batch_size=batch_size, shuffle=False)

    hard_labels = [] 
    X = [] 
    classification_y=[] 

    original_model = original_model.to(device)
    original_model.eval() 

    for step, (X_vector, Y_vector) in enumerate(distill_loader_noShuffle):
        
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output = original_model(X_vector)
        out_y = output.detach().cpu()
        
        softmax_y = softmax(out_y.numpy())
        

        max_indices = np.argmax(softmax_y, axis=1)  
        pre_vector_onehot = F.one_hot(torch.from_numpy(max_indices),n_out)  
        
        X_vector = X_vector.detach().cpu()
        Y_vector = Y_vector.detach().cpu()

        hard_labels.append(pre_vector_onehot)  
        X.append(X_vector)  
        classification_y.append(Y_vector) 
               
    hard_labels = np.vstack(hard_labels)   
    X = np.concatenate(X)   
    classification_y = np.concatenate(classification_y) #(20000,)

    
    hard_labels = hard_labels.astype('float32')
    X = X.astype('float32')
    classification_y = classification_y.astype('int32')



    if(datasetFlag=='CIFAR10' or datasetFlag=='CIFAR100' or datasetFlag=='CINIC10' or datasetFlag=='GTSRB'):
        
        distill_data_with_softlabels = models.CIFARDataForDistill(X, classification_y, hard_labels)
        distill_loader_with_softlabels = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=True)
        distill_loader_with_softlabels_noshuffle = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=False)

        


    elif(datasetFlag=='News' or datasetFlag=='Purchase100'):
        
        distill_data_with_softlabels = models.NewsDataForDistill(X, classification_y, hard_labels)
        distill_loader_with_softlabels = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=True)
        distill_loader_with_softlabels_noshuffle = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=False)
    


    
    params = {}
    if(datasetFlag=='CIFAR10'):
        params['task'] = 'cifar10'
        params['input_size'] = 32
        params['num_classes'] = 10
    elif(datasetFlag=='CIFAR100'):
        params['task'] = 'cifar100'
        params['input_size'] = 32
        params['num_classes'] = 100
    elif(datasetFlag=='CINIC10'):
        params['task'] = 'cinic10'
        params['input_size'] = 32
        params['num_classes'] = 10
    elif(datasetFlag=='GTSRB'):
        params['task'] = 'gtsrb'
        params['input_size'] = 32
        params['num_classes'] = 43
    
    if modelType == 'cnn':
        print('Using a multilayer convolution neural network based model as ref model...')
        net = models.CNNNet(n_hidden=n_hidden, n_out=n_out)
        net = net.to(device)
    elif modelType == 'nn':
        n_in = X.shape[1]
        print('Using a multilayer neural network based model as ref model...')
        net = models.NNNet(n_in, n_hidden, n_out) #n_hidden=128
        net = net.to(device)
    elif modelType =='vgg':
        print('Using vgg model as ref model...')
        
        params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        params['fc_layers'] = [512, 512]
        params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        params['conv_batch_norm'] = True
        params['init_weights'] = True   
        params['augment_training'] = True
        net = models.VGG(params)
        net = net.to(device)
    elif modelType == 'resnet':
        print('Using resnet model as ref model...')
        params['block_type'] = 'basic'
        params['num_blocks'] = [9,9,9]    
        params['augment_training'] = True 
        params['init_weights'] = True  
        net = models.ResNet(params)
        net = net.to(device)
    elif modelType == 'mobilenet':
        print('Using mobilenet model as ref model...')
        params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        params['augment_training'] = True
        net = models.MobileNet(params)
        net = net.to(device)


    net.train()
    criterion = nn.CrossEntropyLoss() 
    criterion = criterion.to(device)

    if  modelType == 'vgg':
        l2_ratio = 0.0005  
        learning_rate = 0.1   

        
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

        
        momentum = 0.9
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_ratio)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9) 

        
        
    elif modelType == 'cnn' or modelType == 'resnet' or modelType == 'mobilenet':
        l2_ratio = 0.0001
        learning_rate = 0.01   
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7) 
    elif modelType == 'nn':
        l2_ratio = 1e-7
        learning_rate = 0.01   
        
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
        
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7) 

    
    print('dataset: {},  model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(datasetFlag, modelType, device, epochs, batch_size, learning_rate, l2_ratio))
    
    count = 1
    print('Training refmodel...')    
    for epoch in range(epochs):
        running_loss = 0

    
        for step, (X_vector, Y_vector, hardlabel_vector) in enumerate(distill_loader_with_softlabels):	
            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)
            hardlabel_vector = hardlabel_vector.to(device)
            
            output = net(X_vector)    
            
            
            
            
            loss = criterion(output, hardlabel_vector)  

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        if optimizer.param_groups[0]['lr']>0.0005:  
            scheduler.step()  
        if (epoch + 1) % 10 == 0: 
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))

    print("Training refmodel finished!")
    
    return net
    

def init_args():
    parser = argparse.ArgumentParser(description='DHAttack Base Params')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--classifierType', type=str, default='mobilenet')
    parser.add_argument('--disturb_num', type=int, default=30)
    parser.add_argument('--num_epoch_for_refmodel', type=int, default=100)
    parser.add_argument('--fixedSampleDataset', type=str, default='GTSRB')
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    

    dataset = args.dataset  
    classifierType = args.classifierType  
    num_epoch_for_refmodel = args.num_epoch_for_refmodel  
    disturb_num = args.disturb_num     
    fixedSampleDataset = args.fixedSampleDataset
    
    
    if dataset == 'CIFAR10':
        pathToLoadData = './data/cifar-10-download'  
    elif dataset == 'CIFAR100':
        pathToLoadData = './data/cifar-100-download'
    elif dataset == 'CINIC10':
        pathToLoadData = './data/CINIC-10-download'
    elif dataset == 'GTSRB':
        pathToLoadData = './data/GTSRB-download'
    elif dataset == 'News':   
        pathToLoadData = None
    elif dataset == 'Purchase100':
        pathToLoadData = './data/Purchase-download'
    else:
        print("Wrong direction! Please see readme.")
    
    dataFolderPath='./data/'  
    

    resultDataPath = './results/'
    label_only_disturb_data = './label_only_disturb_data/' + dataset + '/' 
    try:
        os.makedirs(label_only_disturb_data)
    except OSError:
        pass
    try:
        os.makedirs(resultDataPath)
    except OSError:
        pass
       

    
    disturbFlag = True 
    batch_size = 100  
    distillFlag = False 
    num_workers = 1  
    refModelNum = 256 

    distilledModelNum = refModelNum
    used_num = distilledModelNum 
    num_epoch_for_distillation = num_epoch_for_refmodel
    directionDataset = fixedSampleDataset

    if dataset == 'CIFAR10' and directionDataset == "GTSRB":
        print("This is the case of Outside.")
    elif dataset == 'CIFAR10' and directionDataset == "CIFAR10":
        print("This is the case of Inside.")
    else:
        print("Please check the dataset setting. The parameter dataset can only be CIFAR10, and fixedSampleDataset can only be CIFAR10 or GTSRB!")
        exit()


    
    start_time = time.time()
  

    
    AttackMethod = "DHAttack"
      
    if(AttackMethod == "DHAttack"): 
        target_classifierType = classifierType   
        print("target model is {}".format(target_classifierType))
        print(dataset)
        print("local model is {}".format(classifierType))
        print(AttackMethod)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataPath = dataFolderPath+dataset +'/PreprocessedData_5_part'
        modelFolderPath = './models/'
        modelPath = modelFolderPath + dataset  
        modelsavepath = modelPath +'/HardLabel/'+ classifierType
        try:
            os.makedirs(modelsavepath)
        except OSError:
            pass

        
        targetModel = torch.load(modelPath + '/targetModel_{}.pkl'.format(target_classifierType))
        
        targetModel = targetModel.to(device)
        


        

        

        
        targetTrain, targetTrainLabel  = load_data_for_trainortest(dataPath + '/targetTrain.npz')
        targetTrain = targetTrain.astype(np.float32)
        targetTrainLabel = targetTrainLabel.astype(np.int32) 

        targetTest, targetTestLabel  = load_data_for_trainortest(dataPath + '/targetTest.npz') 
        targetTest = targetTest.astype(np.float32)
        targetTestLabel = targetTestLabel.astype(np.int32)

        n_out = len(np.unique(targetTrainLabel))  

        if disturbFlag == True:
            
            direction_data, direction_label  = load_data_for_trainortest(dataFolderPath + directionDataset + '/PreprocessedData_5_part' + '/targetTrain.npz')
            direction_data = direction_data.astype(np.float32)
            direction_label = direction_label.astype(np.int32)
            random_number1 = random.randint(0, len(direction_data))
            print("The direction point ID is {}  from  {}".format(random_number1, directionDataset))
            direction = direction_data[random_number1]


            
            del direction_data
            del direction_label

            
            index_of_samples = []
            groundtruth_of_samples = []
            sample_num, ch, row, col  = targetTrain.shape
            noisy_images = np.zeros((sample_num, disturb_num, ch, row, col), dtype=np.float32)  #(10000, 300, 3, 32, 32)
            print('Add Mask to targetTrain!')
            for i in tqdm(range(len(targetTrain))):
                for j in range(disturb_num):
                    index_of_samples.append(i)  
                    groundtruth_of_samples.append(targetTrainLabel[i]) 
                    level = j/disturb_num    
                    noisy_images[i, j] = add_mask_bydirection(targetTrain[i],level,direction)
                    

            noisy_images = noisy_images.reshape((sample_num*disturb_num, 3, 32, 32))
            index_of_samples = np.array(index_of_samples)
            groundtruth_of_samples = np.array(groundtruth_of_samples)

            np.savez(label_only_disturb_data + 'disturbData_targetTrain.npz', index_of_samples, noisy_images, groundtruth_of_samples) 

            
            index_of_samples = []
            groundtruth_of_samples = []
            sample_num, ch, row, col  = targetTest.shape
            noisy_images = np.zeros((sample_num, disturb_num, ch, row, col), dtype=np.float32)  
            print('Add Mask to targetTest!')
            for i in tqdm(range(len(targetTest))):
                for j in range(disturb_num):
                    index_of_samples.append(i)  
                    groundtruth_of_samples.append(targetTestLabel[i]) 
                    level = j/disturb_num    
                    noisy_images[i, j] = add_mask_bydirection(targetTest[i],level,direction)
                    

            noisy_images = noisy_images.reshape((sample_num*disturb_num, 3, 32, 32))
            index_of_samples = np.array(index_of_samples)
            groundtruth_of_samples = np.array(groundtruth_of_samples)

            np.savez(label_only_disturb_data + 'disturbData_targetTest.npz', index_of_samples, noisy_images, groundtruth_of_samples)  
        
        
        if distillFlag == True:
            for i in range(distilledModelNum):
                print('creating ref_model {} ................................................'.format(i))
                distilledModel = DistillModelUsingHardLabel(targetModel, dataset, num_epoch_for_distillation, classifierType =classifierType, modelFolderPath = './models/', TargetOrShadow='Target')
                torch.save(distilledModel, modelsavepath +'/distilledModel_{}.pkl'.format(i))

        
        refModeleNames =  get_file_name(modelsavepath)  
        refModeles = []
        for i in refModeleNames[:used_num]:
            print("loading ref_model {} from {}".format(i, modelsavepath))
            refModeles.append(torch.load(modelsavepath + '/' + i))             


        
        target_model = targetModel.eval()
        
        
        with np.load(label_only_disturb_data + 'disturbData_targetTrain.npz') as f:
            noisy_index_of_samples_targetTrain, noisy_images_targetTrain, noisy_groundtruth_of_samples_targetTrain = [f['arr_%d' % i] for i in range(len(f.files))]
            
        target_data = models.CIFARData(targetTrain, targetTrainLabel)
        my_loader1 = DataLoader(target_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  #shuffle=True, num_workers=num_workers, pin_memory=True
            
        noise_target_data = models.CIFARData(noisy_images_targetTrain, noisy_groundtruth_of_samples_targetTrain)
        my_loader2 = DataLoader(noise_target_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  #shuffle=True, num_workers=num_workers, pin_memory=True
        print('Loading targetTrain original data and noisyData!OK!')
        
        
        targetTrain_pred_y = []
        for step, (X_vector, Y_vector) in enumerate(my_loader1):  
            
            X_vector = X_vector.to(device)  #float32
            Y_vector = Y_vector.to(device)  #int64

            output = target_model(X_vector)
            out_y = output.detach().cpu() 
            targetTrain_pred_y.append(np.argmax(out_y, axis=1))  
        
        targetTrain_pred_y = np.concatenate(targetTrain_pred_y)  
        print('Accuracy of targetTrain: {}'.format(accuracy_score(targetTrainLabel, targetTrain_pred_y)))  

        targetTrain_pred_y_noisy = []
        for step, (X_vector, Y_vector) in enumerate(my_loader2):	
            #Y_vector = Y_vector.long()
            X_vector = X_vector.to(device)  
            Y_vector = Y_vector.to(device)

            output = target_model(X_vector)
            out_y = output.detach().cpu() 
            targetTrain_pred_y_noisy.append(np.argmax(out_y, axis=1))  
        
        targetTrain_pred_y_noisy = np.concatenate(targetTrain_pred_y_noisy)  
        print('Accuracy of noisy_targetTrain: {}'.format(accuracy_score(noisy_groundtruth_of_samples_targetTrain, targetTrain_pred_y_noisy)))  
        

        
        count = 0
        targetTrain_pred_y_noisy_for_AllOut = []
        for ref in refModeles:
            ref = ref.to(device)
            ref.eval()
            count = count + 1

            targetTrain_pred_y_noisy_forOut = []
            for step, (X_vector, Y_vector) in enumerate(my_loader2):	
                
                X_vector = X_vector.to(device)  
                Y_vector = Y_vector.to(device)

                output = ref(X_vector)
                out_y = output.detach().cpu() 
                targetTrain_pred_y_noisy_forOut.append(np.argmax(out_y, axis=1))  
        
            targetTrain_pred_y_noisy_forOut = np.concatenate(targetTrain_pred_y_noisy_forOut)  
            print('Accuracy of noisy_targetTrain on refModele {}: {}'.format(count,accuracy_score(noisy_groundtruth_of_samples_targetTrain, targetTrain_pred_y_noisy_forOut)))  
            targetTrain_pred_y_noisy_for_AllOut.append(targetTrain_pred_y_noisy_forOut)
        
        
        del noisy_images_targetTrain

        
        
        with np.load(label_only_disturb_data + 'disturbData_targetTest.npz') as f:
            noisy_index_of_samples_targetTest, noisy_images_targetTest, noisy_groundtruth_of_samples_targetTest = [f['arr_%d' % i] for i in range(len(f.files))]
            
        target_data = models.CIFARData(targetTest, targetTestLabel)
        my_loader1 = DataLoader(target_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  #shuffle=True, num_workers=num_workers, pin_memory=True
            
        noise_target_data = models.CIFARData(noisy_images_targetTest, noisy_groundtruth_of_samples_targetTest)
        my_loader2 = DataLoader(noise_target_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  #shuffle=True, num_workers=num_workers, pin_memory=True        
        print('Loading targetTest original data and noisyData!OK!')

        
        targetTest_pred_y = []
        for step, (X_vector, Y_vector) in enumerate(my_loader1):	
            #Y_vector = Y_vector.long()
            X_vector = X_vector.to(device)  #float32
            Y_vector = Y_vector.to(device)  #int64

            output = target_model(X_vector)
            out_y = output.detach().cpu() 
            targetTest_pred_y.append(np.argmax(out_y, axis=1))  
        
        targetTest_pred_y = np.concatenate(targetTest_pred_y)  
        print('Accuracy of targetTest: {}'.format(accuracy_score(targetTestLabel, targetTest_pred_y)))  

        targetTest_pred_y_noisy = []
        for step, (X_vector, Y_vector) in enumerate(my_loader2):	
            #Y_vector = Y_vector.long()
            X_vector = X_vector.to(device)  
            Y_vector = Y_vector.to(device)

            output = target_model(X_vector)
            out_y = output.detach().cpu() 
            targetTest_pred_y_noisy.append(np.argmax(out_y, axis=1))  
        
        targetTest_pred_y_noisy = np.concatenate(targetTest_pred_y_noisy)  
        print('Accuracy of noisy_targetTest: {}'.format(accuracy_score(noisy_groundtruth_of_samples_targetTest, targetTest_pred_y_noisy)))  
        
        
        count = 0
        targetTest_pred_y_noisy_for_AllOut = []
        for ref in refModeles:
            
            count = count + 1

            targetTest_pred_y_noisy_forOut = []
            for step, (X_vector, Y_vector) in enumerate(my_loader2):	
                
                X_vector = X_vector.to(device)  
                Y_vector = Y_vector.to(device)

                output = ref(X_vector)  
                out_y = output.detach().cpu() 
                targetTest_pred_y_noisy_forOut.append(np.argmax(out_y, axis=1))  
        
            targetTest_pred_y_noisy_forOut = np.concatenate(targetTest_pred_y_noisy_forOut)  
            print('Accuracy of noisy_targetTest on refModel {}: {}'.format(count,accuracy_score(noisy_groundtruth_of_samples_targetTest, targetTest_pred_y_noisy_forOut))) 
            targetTest_pred_y_noisy_for_AllOut.append(targetTest_pred_y_noisy_forOut)

        
        del noisy_images_targetTest

        
        count = 0
        for i in range(len(noisy_index_of_samples_targetTrain)):
            if noisy_index_of_samples_targetTrain[i]!=noisy_index_of_samples_targetTrain[i+1]:
                count = count+1 
                break
            else:
                count = count+1
        print("noisy_sample_num is {}".format(count))
        
        boundary_distance_targetTrain = []
        print("members")
        for i in np.unique(noisy_index_of_samples_targetTrain):
                        
            tmp_pred_y_noisy = targetTrain_pred_y_noisy[i*count:i*count+count] 
            
            tmp_noisy_groundtruth_of_samples_targetTrain = noisy_groundtruth_of_samples_targetTrain[i*count:i*count+count]

            
            distance = computeLDistance(tmp_pred_y_noisy,tmp_noisy_groundtruth_of_samples_targetTrain)
            
            boundary_distance_targetTrain.append(distance)
        
        boundary_distance_targetTest = []
        print("non-members")
        for i in np.unique(noisy_index_of_samples_targetTest):
                        
            tmp_pred_y_noisy = targetTest_pred_y_noisy[i*count:i*count+count] 
            
            tmp_noisy_groundtruth_of_samples_targetTest = noisy_groundtruth_of_samples_targetTest[i*count:i*count+count] 

             
            distance = computeLDistance(tmp_pred_y_noisy,tmp_noisy_groundtruth_of_samples_targetTest)          
            
            boundary_distance_targetTest.append(distance)


        
        boundary_distance_targetTrain_for_AllOut = []
        for r in targetTrain_pred_y_noisy_for_AllOut:
            boundary_distance_targetTrain_forOut = []
            for i in np.unique(noisy_index_of_samples_targetTrain):
                       
                tmp_pred_y_noisy = r[i*count:i*count+count]  
                
                tmp_noisy_groundtruth_of_samples_targetTrain = noisy_groundtruth_of_samples_targetTrain[i*count:i*count+count]

                
                distance = computeLDistance(tmp_pred_y_noisy,tmp_noisy_groundtruth_of_samples_targetTrain)
                boundary_distance_targetTrain_forOut.append(distance)

            boundary_distance_targetTrain_forOut = np.array(boundary_distance_targetTrain_forOut)
            boundary_distance_targetTrain_for_AllOut.append(boundary_distance_targetTrain_forOut)        
        print(len(boundary_distance_targetTrain_for_AllOut))  
        
        boundary_distance_targetTrain_for_AllOut = np.vstack(boundary_distance_targetTrain_for_AllOut)  
        boundary_distance_targetTrain_for_AllOut = np.swapaxes(boundary_distance_targetTrain_for_AllOut, 0, 1)  
        
        boundary_distance_targetTest_for_AllOut = []
        for r in targetTest_pred_y_noisy_for_AllOut:
            boundary_distance_targetTest_forOut = []
            for i in np.unique(noisy_index_of_samples_targetTest):
            
                           
                tmp_pred_y_noisy = r[i*count:i*count+count]  
                
                tmp_noisy_groundtruth_of_samples_targetTest = noisy_groundtruth_of_samples_targetTest[i*count:i*count+count]            
                
                
                distance = computeLDistance(tmp_pred_y_noisy,tmp_noisy_groundtruth_of_samples_targetTest)
                boundary_distance_targetTest_forOut.append(distance)
                
            boundary_distance_targetTest_forOut = np.array(boundary_distance_targetTest_forOut)
            boundary_distance_targetTest_for_AllOut.append(boundary_distance_targetTest_forOut)
        print(len(boundary_distance_targetTest_for_AllOut))
        
        boundary_distance_targetTest_for_AllOut = np.vstack(boundary_distance_targetTest_for_AllOut)  #(32,10000)
        boundary_distance_targetTest_for_AllOut = np.swapaxes(boundary_distance_targetTest_for_AllOut, 0, 1) #(10000,32)

    
        
        EvalScores_disModel = np.concatenate((boundary_distance_targetTrain_for_AllOut, boundary_distance_targetTest_for_AllOut), axis=0)

        
        EvalScores_disModel = np.exp(-EvalScores_disModel)  
        
        EvalScores_disModel = EvalScores_disModel + 1e-8
        bscores = EvalScores_disModel
        dat_reference_or_distill = np.log(
            np.exp(-bscores) / (1 - np.exp(-bscores))
        )
        
        mean_reference_or_distill = np.mean(dat_reference_or_distill, 1)           
        
        std_reference_or_distill = np.std(dat_reference_or_distill, 1)
        
        
        membershiplabel_targetTrain = np.ones(len(boundary_distance_targetTrain))
        membershiplabel_targetTest = np.zeros(len(boundary_distance_targetTest))
        membershiplabels = np.concatenate([membershiplabel_targetTrain, membershiplabel_targetTest])
        
        original_EvalScores_Target = np.concatenate((boundary_distance_targetTrain, boundary_distance_targetTest))
        
        
        EvalScores_Target = np.exp(-original_EvalScores_Target)
        EvalScores_Target = EvalScores_Target + 1e-8
        
        prediction = []
        answers = []
        distanceScores = []
        oridistance = []
        
        newEvalScores_Target = np.log(np.exp(-EvalScores_Target)/(1-np.exp(-EvalScores_Target)))
               
        for sc_target, mean_dis, std_dis, ans, ori in zip(newEvalScores_Target, mean_reference_or_distill, std_reference_or_distill, membershiplabels, original_EvalScores_Target):
            score = 1 - scipy.stats.norm.cdf(sc_target, mean_dis, std_dis + 1e-30)   
            
            
            prediction.append(score)   
            answers.append(ans)  
            distanceScores.append(sc_target)
            oridistance.append(ori)
        
        prediction = np.array(prediction)
        prediction = prediction.astype('float32')
        prediction = -prediction   

        answers = np.array(answers)
        answers = answers.astype('int32')

        distanceScores = np.array(distanceScores)
        distanceScores = distanceScores.astype('float32')

        oridistance = np.array(oridistance)
        oridistance = oridistance.astype('float32')

        
        print("Results:")
        print(dataset)
        print(classifierType)
        print(AttackMethod)
        
        print(disturb_num)
        print(used_num)
        np.savez(resultDataPath + '{}_{}_{}_disturb_num{}_used_num{}.npz'.format(AttackMethod,dataset,classifierType,disturb_num,used_num), answers, prediction)  #保存的格式，第一个数据集MemberLabels是真实的member标签，第二个数据集是预测的member标签或概率。
        np.savez(resultDataPath + '{}_{}_{}_disturb_num{}_used_num{}forAna.npz'.format(AttackMethod,dataset,classifierType,disturb_num,used_num), answers, prediction, distanceScores, oridistance)

        
        end_time = time.time()

        
        
        tpr_0, final_auc = ROC_AUC_Result_logshow(answers,prediction,reverse=False)
        
        



    
    time_interval = (end_time - start_time) / 60

    print(f"It took {time_interval:.2f} minutes.")
