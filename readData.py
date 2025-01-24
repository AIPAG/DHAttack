import pickle
from tkinter import Y
import numpy as np
import os
from PIL import Image

import pandas as pd





def readCIFAR10(data_path):    

    print(data_path)   
    

    
    for i in range(5):
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')  
        
        train_data_dict = pickle.load(f, encoding='iso-8859-1')  
        f.close()
        if i == 0:
            
            X = train_data_dict["data"]   
            y = train_data_dict["labels"]
            
            continue
        X = np.concatenate((X , train_data_dict["data"]),   axis=0)  
        y = np.concatenate((y , train_data_dict["labels"]), axis=0)

    
    f = open(data_path + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')  
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])

    return X, y, XTest, yTest


def readCIFAR100(data_path):    

    print(data_path)   

    
    f = open(data_path + '/train', 'rb')  
    
    train_data_dict = pickle.load(f, encoding='iso-8859-1')  
    f.close()
    print(train_data_dict.keys())  
    X = train_data_dict["data"]      
    y = train_data_dict["fine_labels"]  
    
    f = open(data_path + '/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')  
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["fine_labels"])

    return X, y, XTest, yTest




def readImgs(img_path, label, intlabel):
    img_path = img_path + '/' +label
    files = os.listdir(img_path)
    allImgData = []
    for i in files:
        img = Image.open(img_path + '/' + i).convert('RGB')  
        
        img_data = np.array(img)  
        
        img_data = img_data.swapaxes(1,2)
        
        img_data = img_data.swapaxes(0,1)
        
        
        img_data = img_data.ravel()
        
        
        allImgData.append(img_data)
    allImgData = np.array(allImgData)
    
    y = np.full((allImgData.shape[0]), intlabel).astype(np.int32)  
    return allImgData, y



def readCINIC10(data_path):    
    
    
    
    
    img_path = os.path.join(data_path+'/train')
    labels_dict = {}   
    files = os.listdir(img_path)
    

    X_data_train = []
    y_data_train = []
    
    count = 0
    for i in files:  
        
        labels_dict[i]=count
        count = count + 1

    for j in labels_dict.keys(): 
        X,y = readImgs(img_path,j,labels_dict[j])  
        X_data_train.append(X)
        y_data_train.append(y)
    
    
    X_data_train = np.concatenate(X_data_train).astype(np.int64)  
    
    y_data_train = np.concatenate(y_data_train).astype(np.int64)
    
     
    
    img_path = os.path.join(data_path+'/test')
    files = os.listdir(img_path)
    X_data_test = []
    y_data_test = []
    for j in labels_dict.keys(): 
        X,y = readImgs(img_path,j,labels_dict[j])  
        X_data_test.append(X)
        y_data_test.append(y)
    
    X_data_test = np.concatenate(X_data_test).astype(np.int64)  
    
    y_data_test = np.concatenate(y_data_test).astype(np.int64)
    

    
    img_path = os.path.join(data_path+'/valid')
    files = os.listdir(img_path)
    X_data_valid = []
    y_data_valid = []
    for j in labels_dict.keys(): 
        X,y = readImgs(img_path,j,labels_dict[j])  
        X_data_valid.append(X)
        y_data_valid.append(y)
    
    X_data_valid = np.concatenate(X_data_valid).astype(np.int64)  
    
    y_data_valid = np.concatenate(y_data_valid).astype(np.int64)
    
    
    return X_data_train, y_data_train, X_data_test, y_data_test, X_data_valid, y_data_valid






def reshape_for_save(raw_data):   
        raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))   
        
        raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3))
        

        
        raw_data = raw_data.transpose(0,3,1,2)  
        
        return raw_data.astype(np.float32)  


def rescale(raw_data,offset,scale):
        newdata =  reshape_for_save(raw_data)
        

        return (newdata - offset) / scale    

def preprocessingCIFAR(toTrainData, toTestData):
    
    if(toTestData.size!=0):
        print("train data size:")
        print(np.shape(toTrainData))   
        print("test data size:")
        print(np.shape(toTestData))   

        newdata = reshape_for_save(toTrainData)
        
        offset = np.mean(newdata, 0) 

        scale  = np.std(newdata, 0).clip(min=1)   

       
        return rescale(toTrainData,offset,scale), rescale(toTestData,offset,scale)  #(10520, 3, 32, 32)
    else:
        print("distillation data size:")
        print(np.shape(toTrainData))   

        newdata = reshape_for_save(toTrainData)
        
        offset = np.mean(newdata, 0)  

        scale  = np.std(newdata, 0).clip(min=1)   

        
        return rescale(toTrainData,offset,scale)

###########################################################################################################




def preprocessingNews(toTrainData, toTestData):  
    def normalizeData(X):   
        offset = np.mean(X, 0)   
        scale = np.std(X, 0).clip(min=1)  
        X = (X - offset) / scale
        X = X.astype(np.float32)	
        return X 
    
    if(toTestData.size!=0):
        return normalizeData(toTrainData),normalizeData(toTestData)
    else:
        return normalizeData(toTrainData)





def readGTSRB(data_path):    
    
    
    img_path = os.path.join(data_path+'/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')  
    files = os.listdir(img_path)
    

    X_data_train = []
    y_data_train = []

    for j in files: 
        X,y = readImgsOfGTSRB(img_path + '/' +j)  
        X_data_train.append(X)
        y_data_train.append(y)
    
    
    X_data_train = np.concatenate(X_data_train).astype(np.int64) 
    
    y_data_train = np.concatenate(y_data_train).astype(np.int64)
    
    X_data_test = []
    y_data_test = []
    img_path = os.path.join(data_path+'/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')
    X,y = readImgsOfGTSRB(img_path)
    X_data_test.append(X)
    y_data_test.append(y)
    
    X_data_test = np.concatenate(X_data_test).astype(np.int64)  
    
    y_data_test = np.concatenate(y_data_test).astype(np.int64)
    
    
    
    return X_data_train, y_data_train, X_data_test, y_data_test   


def readImgsOfGTSRB(img_path):
    files = os.listdir(img_path)   
    allImgData = []
    y = []
    for f in files:
        
        if f.endswith(".csv"):
            csv_data = pd.read_csv(img_path + '/' + f)
    
    imgDesc = np.array(csv_data)  

    
    for i in range(imgDesc.shape[0]):  
        img_info = np.array(imgDesc)[i,:].tolist()[0].split(";")   
        
        img_name = img_info[0]
        img = Image.open(img_path + '/' + img_name).convert('RGB')  
        box = (int(img_info[3]),int(img_info[4]),int(img_info[5]),int(img_info[6]))
        img_roi = img.crop(box)  
        newimg = img_roi.resize((32,32),Image.ANTIALIAS)   
        
        img_data = np.array(newimg)  
        
        
        img_data = img_data.swapaxes(1,2)
        
        img_data = img_data.swapaxes(0,1)
        
        img_data = img_data.ravel()
        
        
        allImgData.append(img_data)
        y.append(img_info[7])
    allImgData = np.array(allImgData)   
    y = np.array(y) 
    
    return allImgData, y





def load_data_for_location(data_path):   
    files = os.listdir(data_path)
    for i in files:    
        
        if i.endswith(".npz"): 
            with np.load(data_path+'/'+i) as f:
                
                train_x = f['x']   #(5010, 446)
                train_y = f['y']   #(5010,)
                   
    return train_x, train_y

def preprocessingLocationLabel(toTrainDataLabel, toTestDataLabel):  
    def normalizeData(X):   	
        return X - 1
    
    if(toTestDataLabel.size!=0):
        return normalizeData(toTrainDataLabel),normalizeData(toTestDataLabel)
    else:
        return normalizeData(toTrainDataLabel)
    





def load_data_for_Purchase(data_path):  
    files = os.listdir(data_path)
    for i in files:    
        
        data_set =np.genfromtxt(data_path+'/'+i,delimiter=',')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        print(X.shape, Y.shape)       
    return X, Y






