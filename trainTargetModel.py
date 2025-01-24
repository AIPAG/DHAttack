import numpy as np
import math

import random
import os
import Models as models  
import readData as rd  

import torch.nn as nn
import torch 
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse


def load_data_for_trainortest(data_name):   
    with np.load( data_name) as f:
        
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))] 
    return train_x, train_y

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

def softmax(x):
   
   shift = np.amax(x, axis=1) 
   shift = shift.reshape(-1,1)  
   x = x - shift    

   exp_values = np.exp(x)
   denominators = np.sum(np.exp(x), axis=1) 
   softmax_values = (exp_values.T/denominators).T  
   return softmax_values

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
    elif(datasetFlag=='News' or datasetFlag=='Location' or datasetFlag=='Purchase100'):
        
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

    elif model == 'wideresnet':
        print('Using wideresnet model...')
        params['block_type'] = 'bottle'
        params['num_blocks'] = [5,5,5]
        params['widen_factor'] = 4
        params['dropout_rate'] = 0.3
        params['augment_training'] = True 
        params['init_weights'] = True
        net = models.WideResNet(params)
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
    classification_y = np.concatenate(classification_y) 

    
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')

    return attack_x, attack_y, net, classification_y



def initializeShadowModel(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './models/',classifierType = 'cnn'):
    
    dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'
    attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
    modelPath = modelFolderPath + dataset
    try:
        os.makedirs(attackerModelDataPath)
    except OSError as ee:
        #print(ee)
        pass
    try:
        os.makedirs(modelPath)
    except OSError as ee:
        #print(ee)
        pass
    print("Training the Shadow model for {} epoch".format(num_epoch))


    
    shadowTrain, shadowTrainLabel  = load_data_for_trainortest(dataPath + '/shadowTrain.npz')
    shadowTest,  shadowTestLabel   = load_data_for_trainortest(dataPath + '/shadowTest.npz')

    
    attackModelDataShadow, attackModelLabelsShadow, shadowModelToStore, classification_y = trainTarget(classifierType,shadowTrain, shadowTrainLabel, X_test=shadowTest, y_test=shadowTestLabel, splitData= False, inepochs=num_epoch, batch_size=100, datasetFlag = dataset) 

    
    torch.save(shadowModelToStore, modelPath + '/shadowModel_{}.pkl'.format(classifierType))
    shadowModelToStore = torch.load(modelPath + '/shadowModel_{}.pkl'.format(classifierType))
    
    return shadowModelToStore   



def initializeTargetModel(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './models/',classifierType = 'cnn'):
    
    dataPath = dataFolderPath+dataset+'/PreprocessedData_5_part'
    attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
    modelPath = modelFolderPath + dataset
    try:
        os.makedirs(attackerModelDataPath)
    except OSError as ee:
        #print(ee)
        pass
    try:
        os.makedirs(modelPath)
    except OSError as ee:
        #print(ee)
        pass
    print("Training the Target model for {} epoch".format(num_epoch))


    
    targetTrain, targetTrainLabel  = load_data_for_trainortest(dataPath + '/targetTrain.npz')
    targetTest,  targetTestLabel   = load_data_for_trainortest(dataPath + '/targetTest.npz')

    
    attackModelDataTarget, attackModelLabelsTarget, targetModelToStore, classification_y = trainTarget(classifierType,targetTrain, targetTrainLabel, X_test=targetTest, y_test=targetTestLabel, splitData= False, inepochs=num_epoch, batch_size=100, datasetFlag = dataset) 

    
    torch.save(targetModelToStore, modelPath + '/targetModel_{}.pkl'.format(classifierType))
    targetModelToStore = torch.load(modelPath + '/targetModel_{}.pkl'.format(classifierType))

    
    return targetModelToStore   

def init_args():
    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='CIFAR10 CIFAR100 CINIC10 GTSRB')
    parser.add_argument('--classifierType', type=str, default='mobilenet',help='vgg, resnet, mobilenet')
    parser.add_argument('--num_epoch', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    

    dataset = args.dataset  
    classifierType = args.classifierType  
    num_epoch = args.num_epoch
    
    
    
    
    if dataset == 'CIFAR10':
        pathToLoadData = './data/cifar-10-download'  
    elif dataset == 'CIFAR100':
        pathToLoadData = './data/cifar-100-download'
    elif dataset == 'CINIC10':
        pathToLoadData = './data/CINIC-10-download'
    elif dataset == 'GTSRB':
        pathToLoadData = './data/GTSRB-download'
    elif dataset == 'News':   #download from net.
        pathToLoadData = None
    elif dataset == 'Purchase100':
        pathToLoadData = './data/Purchase-download'
    else:
        print("Wrong direction! Please see readme.")

    dataFolderPath='./data/'  

    
    
    


    targetModel = initializeTargetModel(dataset,num_epoch,classifierType =classifierType)


    shadowModel = initializeShadowModel(dataset,num_epoch,classifierType =classifierType)
