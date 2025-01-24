from re import X
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset  
import numpy as np
import torch.nn.functional as F
import math

import torch.nn.utils.rnn as rnn_utils



class CNNNet(nn.Module):
    def __init__(self,n_hidden, n_out):
        super(CNNNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=1, padding=2)
        
        self.relu1 = nn.ReLU() 

       
        self.pool1 = nn.MaxPool2d(2)  

        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5), stride=1, padding=0)  
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) 

        
        self.fc1 = nn.Linear(1152,n_hidden) 
        self.tanh = nn.Tanh() 

        
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.softmax = nn.Softmax()  


    
    def forward(self,x):
        
        x = self.conv1(x)  #torch.Size([100, 32, 32, 32])
        x = self.relu1(x)  #torch.Size([100, 32, 32, 32])
        x = self.pool1(x)  #torch.Size([100, 32, 16, 16])

        x = self.conv2(x)  #torch.Size([100, 32, 12, 12])
        x = self.relu2(x)  #torch.Size([100, 32, 12, 12])
        x = self.pool2(x)  #torch.Size([100, 32, 6, 6])

        
        
        x = x.view(-1, 1152)  #torch.Size([100, 1152])
        x = self.fc1(x)  #torch.Size([100, 128])  
        x = self.tanh(x) #torch.Size([100, 128])

        x = self.fc2(x)  #torch.Size([100, 10])

        

        return x



class NNNet(nn.Module):
    def __init__(self,n_in,n_hidden, n_out):
        super(NNNet,self).__init__()
        self.fc1 = nn.Linear(n_in,n_hidden) 
        self.tanh1 = nn.Tanh() 
        self.fc2 = nn.Linear(n_hidden,n_out) 
        


    
    def forward(self,x):
        
        x = self.fc1(x)
        x = self.tanh1(x) 
        x = self.fc2(x)
        

        return x




class SoftmaxModel(nn.Module):
    def __init__(self,n_in, n_out):
        super(SoftmaxModel,self).__init__()

        self.fc1 = nn.Linear(n_in,64) 
        self.tanh = nn.Tanh() 
        self.fc2 = nn.Linear(64,n_out) 


    
    def forward(self,x):
        
        x1 = self.fc1(x)
        x2 = self.tanh(x1) 
        x = self.fc2(x2)
        

        return x, x2

class MLP4LayerModel(nn.Module):
    def __init__(self,n_in, n_out):
        super(MLP4LayerModel,self).__init__()
        
        self.relu = nn.ReLU() 
        self.fc1 = nn.Linear(n_in,512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, n_out)


    
    def forward(self,x):
        
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.relu(x) 
        x = self.fc3(x)
        x2 = self.relu(x)
        x = self.fc4(x2)   

        return x, x2






class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))        
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))
                
        conv_layers.append(nn.ReLU())
                
        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))
        
        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class VGG(nn.Module):
    def __init__(self, params):
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels'] 
        self.fc_layer_sizes = params['fc_layers']

        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        
        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel
        
        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True
            
            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width
        
        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

##################### ResNet  ############
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        
        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers
            
    def forward(self, x):
        fwd = self.layers[0](x) 
        fwd += self.layers[1](x) 
        fwd = self.layers[2](fwd) 
        return fwd

class ResNet(nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']

        self.in_channels = 16
        self.num_output =  1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))
        
        end_layers = []

        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

        self.augment_training = params['augment_training']

        

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.init_conv(x)
        
        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


############################## MobileNet   #################
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class MobileNet(nn.Module):
    def __init__(self, params):
        super(MobileNet, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])

        self.num_output = 1
        self.in_channels = 32
        init_conv = []
        
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []

        end_layers.append(nn.AvgPool2d(2))

        end_layers.append(Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd










class CIFARData(Dataset): 
    def __init__(self, X_train, y_train):
        self.X_train = X_train  
        self.y_train = y_train  

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]

        
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label


class CIFARDataForDistill(Dataset): 
    def __init__(self, X_train, y_train, softlabel):
        self.X_train = X_train  
        self.y_train = y_train  
        self.softlabel = softlabel 

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        softlabel = self.softlabel[idx]

         
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        
        return img, label, softlabel


class NewsData(Dataset): 
    def __init__(self, X_train, y_train):
        self.X_train = X_train  
        self.y_train = y_train  

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.X_train[idx]
        label = self.y_train[idx]

         
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return x, label


class NewsDataForDistill(Dataset): 
    def __init__(self, X_train, y_train, softlabel):
        self.X_train = X_train  #X_train.shape      (4500, 134410)
        self.y_train = y_train  #(4500,)
        self.softlabel = softlabel 

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.X_train[idx]
        label = self.y_train[idx]
        softlabel = self.softlabel[idx]

        
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return x, label, softlabel









class CELossUsingSoftLabels(nn.Module): 
    def __init__(self):
        super().__init__()
        return
    
    def forward(self,softlabels, pre_vectors):
        CElosses =  -torch.sum(softlabels * F.log_softmax(pre_vectors,dim=1), dim=1)
        loss = torch.sum(CElosses)/len(pre_vectors)
        return loss





