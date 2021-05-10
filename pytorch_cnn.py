#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:41:19 2021
ccc
@author: xavier_wong
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth = 120)

train_set = torchvision.datasets.FashionMNIST(
    root = "./data/FashionMNIST"
    ,train = True
    ,download = True
    ,transform = transforms.Compose([
        transforms.ToTensor()
        ])
    )

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size= 5)
        self.conv2 = nn.Conv2d(in_channels= 6, out_channels= 12, kernel_size= 5)
        
        self.fc1 = nn.Linear(in_features= 12*4*4,out_features= 120)
        self.fc2 = nn.Linear(in_features = 120,out_features=60)
        self.out = nn.Linear(in_features= 60, out_features=10)
    
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size = 2,stride = 2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size = 2,stride = 2)
        
        t = F.relu(self.fc1(t.reshape(-1,12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        
        return t

##关闭自动计算图
torch.set_grad_enabled(False)

network = Network()
sample = next(iter(train_set))

image,label = sample 
image.shape 

image.unsqueeze(0).shape

pred = network(image.unsqueeze(0))
pred.argmax(dim = 1)

##########
import torch.optim as optim

torch.set_grad_enabled(True)


###############单次训练
network = Network()

train_loader = torch.utils.data.DataLoader(
    train_set
    ,batch_size = 100
    )

##采用的Adam优化
optimizer = optim.Adam(network.parameters(),lr = 0.01)

batch = next(iter(train_loader))
images, labels = batch

preds = network(images)
loss = F.cross_entropy(preds,labels)  ##交叉熵

##反向传播求导
loss.backward()   ##计算梯度
optimizer.step()  ##更新权重

print("loss1:",loss.item())
preds = network(images)
loss = F.cross_entropy(preds,labels)
print("loss2:",loss.item())


def get_num_correct(preds,labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()


###############增加循环训练
network = Network()
train_loader = torch.utils.data.DataLoader(
    train_set
    ,batch_size = 100
    )
optimizer = optim.Adam(network.parameters(),lr = 0.01)

for epoch in range(3):
    total_loss = 0
    total_correct = 0
    ##采用的Adam优化
    
    for batch in train_loader:
        images, labels = batch
        
        preds = network(images)
        loss = F.cross_entropy(preds,labels)
        
        optimizer.zero_grad() #梯度归零
        loss.backward()
        optimizer.step()
        
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
        
    print("epoch:", epoch, "total_correct:",total_correct,"loss",total_loss)


total_correct/len(train_set)

######为整个训练集预测，。。建立混淆矩阵

def get_all_preds(model,loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        
        preds = model(images)
        all_preds = torch.cat(
            (all_preds,preds)
            ,dim = 0
            )
    return all_preds

#####这种方法耗费的梯度，训练完做预测不需要梯度
prediction_loader = torch.utils.data.DataLoader(train_set
                                                ,batch_size = 10000)
train_preds = get_all_preds(network,prediction_loader)

train_preds.shape
##检查是否打开了梯度计算图
print(train_preds.requires_grad)
train_preds.requires_grad  = False

train_preds.grad
train_preds.grad_fn

#####利用局部方法，局部关闭梯度，因为预测时不用梯度，节省内存
with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set,batch_size = 10000)
    train_preds = get_all_preds(network, prediction_loader)
    
    
stacked = torch.stack(
    (
     train_set.targets
     , train_preds.argmax(dim = 1)
     )
    ,dim = 1
    )
stacked[0].tolist()

cmt = torch.zeros(10,10,dtype = torch.int64)

for p in stacked:
    tl,pl = p.tolist()
    cmt[tl,pl] = cmt[tl,pl] + 1

#####画出混淆矩阵
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

cm = confusion_matrix(train_set.targets,train_preds.argmax(dim=1))
print(type(cm))
cm 

names = ("TST","T","P","D","S","S1","SN","BA","A")
plt.figure(figsize = (10,10))
plot_confusion_matrix(cm, names)


################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True)

from torch.utils.tensorboard import SummaryWriter


def get_num_correct(preds,labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size= 5)
        self.conv2 = nn.Conv2d(in_channels= 6, out_channels= 12, kernel_size= 5)
        
        self.fc1 = nn.Linear(in_features= 12*4*4,out_features= 120)
        self.fc2 = nn.Linear(in_features = 120,out_features=60)
        self.out = nn.Linear(in_features= 60, out_features=10)
    
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size = 2,stride = 2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size = 2,stride = 2)
        
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        
        t = F.relu(self.fc2(t))
        
        t = self.out(t)
        
        return t


train_set = torchvision.datasets.FashionMNIST(
    root = "./data"
    ,train = True
    ,download = True
    ,transform = transforms.Compose([
        transforms.ToTensor()
        ])
    )



network = Network()
train_loader = torch.utils.data.DataLoader(
    train_set
    ,batch_size = 100
    )
optimizer = optim.Adam(network.parameters(),lr = 0.01)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

####starting with TensorBoard
tb = SummaryWriter()
tb.add_image("images", grid)
tb.add_graph(network,images)


#####the training loop review

for epoch in range(5):
    
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader:
        images,labels = batch
        
        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
        
    tb.add_scalar("Loss",total_loss,epoch)
    tb.add_scalar("Number Correct",total_correct,epoch)
    tb.add_scalar("Accuracy", total_correct/len(train_set),epoch)
    
    tb.add_histogram("conv1.bias",network.conv1.bias,epoch)
    tb.add_histogram("conv1.weight",network.conv2.weight,epoch)
    tb.add_histogram("conv1.weight.grad",network.conv1.weight.grad,epoch)
    
    for name,weight in network.named_parameters():
        tb.add_histogram(name,weight,epoch)
        tb.add_histogram(f"{name}.grad",weight.grad,epoch)
    print("epoch:", epoch, "total_correct:",total_correct,"loss",total_loss)

tb.close()

##############调参
from itertools import product

parameters = dict(
    lr = [0.01,0.001]
    ,batch_size = [10,100,1000]
    ,shuffle = [True,False]
    )

param_values = [v for v in parameters.values()]

for lr,batch_size,shuffle in product(*param_values):
  
    network = Network()
    
    train_loader = torch.utils.data.DataLoader(
        train_set
        ,batch_size = batch_size
        ,shuffle = shuffle
        )
    optimizer = optim.Adam(network.parameters(),lr = lr)
    
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    
    ####starting with TensorBoard
    
    comment = f"lr = {lr} batch_size= {batch_size} shuffle={shuffle}"
    tb = SummaryWriter(comment = comment)
    tb.add_image("images", grid)
    tb.add_graph(network,images)

    for epoch in range(5):
        
        total_loss = 0
        total_correct = 0
        
        for batch in train_loader:
            images,labels = batch
            
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()*batch_size
            ###注意这里是*batch_size,是为了转化比较
            total_correct += get_num_correct(preds,labels)
            
        tb.add_scalar("Loss",total_loss,epoch)
        tb.add_scalar("Number Correct",total_correct,epoch)
        tb.add_scalar("Accuracy", total_correct/len(train_set),epoch)
        
        # tb.add_histogram("conv1.bias",network.conv1.bias,epoch)
        # tb.add_histogram("conv1.weight",network.conv2.weight,epoch)
        # tb.add_histogram("conv1.weight.grad",network.conv1.weight.grad,epoch)
        
        for name,weight in network.named_parameters():
            tb.add_histogram(name,weight,epoch)
            tb.add_histogram(f"{name}.grad",weight.grad,epoch)
        print("epoch:", epoch, "total_correct:",total_correct,"loss",total_loss)

tb.close()    
    


#####下面是另一种生成参数笛卡尔积，进行循环的方法
###这种方法更容易扩展，更容易推理
from collections import OrderedDict
from collections import namedtuple
from itertools import product


class RunBuilder():
    @staticmethod   ##不需要类的实例来调用
    def get_runs(params):
        
        Run = namedtuple("Run", params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs

#############
import time
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment = f"-{run}")
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image("images",grid)
        self.tb.add_graph(self.network,images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0 
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
    
    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar("Loss", loss, self.epoch_count)
        self.tb.add_scalar("Accuracy", accuracy,self.epoch_count)
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f"name.grad", param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        
        for k,v in self.run_params._asdict().items():
            results[k] = v
            
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data,orient = "columns")
        
        # display.clear_output(wait = True)
        print(df)
            
    def track_loss(self,loss):
        self.epoch_loss += loss.item() * self.loader.batch_size
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds,labels)
        
    @torch.no_grad()
    def _get_num_correct(self,preds,labels):
        return preds.argmax(dim =1).eq(labels).sum().item()
    
    def save(self,fileName):
        pd.DataFrame.from_dict(
            self.run_data
            ,orient = "columns"
            ).to_csv(f"{fileName}.csv")
        
        with open(f"{fileName}.json","w",encoding = "utf-8") as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent = 4)
            

#######另一个文档
            
params = OrderedDict(
    lr = [0.01,0.001]
    , batch_size = [1000,2000]
    , shuffle = [True,False]
    )   
        

m = RunManager()

for run in RunBuilder.get_runs(params):
    
    network = Network()
    
    loader = torch.utils.data.DataLoader(
        train_set
        ,batch_size = run.batch_size
        ,shuffle = run.shuffle
        )
    optimizer = optim.Adam(network.parameters(),lr = run.lr)
    
    m.begin_run(run,network,loader)
    for epoch in range(1):
        m.begin_epoch()
        for batch in loader:
            
            images,labels = batch
            
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            m.track_loss(loss)
            m.track_num_correct(preds,labels)
        m.end_epoch()
    m.end_run()
m.save("/Users/xavier_wong/Documents/Python Learning/results")











 
