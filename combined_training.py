import numpy as np

from torch import from_numpy
from torch import cuda
from torch import no_grad
from torch import optim

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

import pandas as pd

#%matplotlib inline

from GNN import *
from SkeletonDataset_alphapose import * #For JAAD
from SkeletonsDataset import * #dataloader for Carla
from ModelTrainEvaluate import *
from MetricsPlots import *
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import RandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
import random
from torch.utils.data import Subset
import os

seed=42
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

info = 32

train_dataset_carla = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/data/carla_joints_using_alphapose/train_carla_Alphapose.csv',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)
#train_dataset_carla.loadedData[['video','frame','decision_point','keypoints','crossing']]

train_dataset_jaad=Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/jaad_ped_graph/Jaad_323/jaad_default_train.csv',numberOfJoints=17,
                                 normalization='minmax', target='cross', info=info )
val_dataset = Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/jaad_ped_graph/Jaad_323/jaad_default_val.csv',numberOfJoints=17, normalization='minmax',
                               target='cross', info=info)
#val_dataset_carla = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/data/val_carla_jaad_final15.csv',
                                 #normalization='minmax', target='crossing', info=info,
                                 #numberOfJoints=15, remove_undetected=False)
#val_dataset_jaad = Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/JAAD/val_jaad_final_17_15.csv',numberOfJoints=15, normalization='minmax',
                               #target='cross', info=info)
combined_dataset_train = ConcatDataset([train_dataset_carla, train_dataset_jaad])
#combined_dataset_val = ConcatDataset([val_dataset_carla, val_dataset_jaad])
#val_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/data/val_carla_jaad_final15.csv',
                                 #normalization='minmax', target='crossing', info=info,
                                 #numberOfJoints=15, remove_undetected=False)


#combined_dataset_train.loadedData[['video','frame','skeleton','crossing']]
#val_dataset.loadedData[['video','frame','keypoints_15','cross']]
#totalRows = len(combined_dataset_train.loadedData)
#crossingRows = len(combined_dataset_train.loadedData[combined_dataset_train.loadedData['crossing']==True])
#nocrossingRows = len(combined_dataset_train.loadedData[combined_dataset_train.loadedData['crossing']!=True])

#print('Training dataset total rows:', totalRows)
#print('Training dataset crossing class samples:', crossingRows)
#print('Training dataset not-crossing class samples:', nocrossingRows)
print('Training len:', len(combined_dataset_train))
totalRows_val = len(val_dataset.loadedData)
crossingRows_val = len(val_dataset.loadedData[val_dataset.loadedData['crossing']==True])
nocrossingRows_val = len(val_dataset.loadedData[val_dataset.loadedData['crossing']!=True])
print('validation dataset total rows:', totalRows_val)
print('validation len:', len(val_dataset))
print('val dataset crossing class samples:', crossingRows_val)
print('val dataset not-crossing class samples:', nocrossingRows_val)

val_dataset.shuffle()
#val_dataset.loadedData[['video','frame','decision_point','keypoints','crossing']]

#Training
numberOfClasses = 2

#y =combined_dataset_train.loadedData['crossing'].to_numpy()
#y = np.where(y==1, 1, 0)
#bc = np.bincount(y)

#class_weights = len(combined_dataset_train.loadedData) / (numberOfClasses * bc)
#class_weights = torch.tensor(class_weights, dtype=torch.float)
#shuffled_dataset = Subset(combined_dataset_train, range(len(combined_dataset_train)))
#random.shuffle(shuffled_dataset)
#print('class_weights:', class_weights)
#combined_dataset_train.shuffle()
#shuffled_dataset = combined_dataset_train[:]
#random.shuffle(shuffled_dataset)
#random.shuffle(combined_dataset_train)
# First element of training subset:

t0 =combined_dataset_train[0]

# Node features:
t1 = t0.x_temporal[0]

# Number of nodes:
numberOfNodes = t1.shape[0]

# Number of dimensions of each node features:
embed_dim = t1.shape[1]
#embed_dim = 2

print('Number of nodes per skeleton:', numberOfNodes)
print('Number of features per node:', embed_dim)
num_epochs = 60
batch_size = 300

device = torch.device('cuda')
model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005) #in original code for carla
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
crit = torch.nn.BCELoss() #weight=class_weights)

#train_loader = DataLoader(train_dataset, batch_size=batch_size)
#val_loader = DataLoader(val_dataset, batch_size=batch_size)
#weights_train = [1/len(train_dataset_carla), 1/len(train_dataset_jaad)]
#sampler_train = WeightedRandomSampler(weights_train, len(combined_dataset_train))
#weights = [len(train_dataset_carla)/(len(train_dataset_jaad)+len(train_dataset_carla)),len(train_dataset_jaad)/(len(train_dataset_jaad)+len(train_dataset_carla))] * (len(train_dataset_jaad) + len(train_dataset_carla))
weights = [len(train_dataset_carla)/(len(train_dataset_jaad)+len(train_dataset_carla)) if i <len(train_dataset_carla) else len(train_dataset_jaad)/(len(train_dataset_jaad)+len(train_dataset_carla)) for i in range(len(train_dataset_jaad)+len(train_dataset_carla))]
sampler = WeightedRandomSampler(weights, len(weights))
#train_loader = DataLoader(combined_dataset_train, batch_size=batch_size, sampler=RandomSampler(combined_dataset_train))
train_loader = DataLoader(combined_dataset_train, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
#weights_val = [1/len(val_dataset_carla), 1/len(train_dataset_jaad)]
#sampler_val = WeightedRandomSampler(weights_val, len(combined_dataset_val))
#val_loader = DataLoader(combined_dataset_train, batch_size=batch_size, sampler=sampler_val)

loss_values = []

metrics_train = []
metrics_val = []
synthetic_counter = 0
real_counter = 0
for i, epoch in enumerate(range(num_epochs)):
    for i, inputs in enumerate(train_loader):
        '''if i < len(train_dataset_carla):
            synthetic_counter += 1
        else:
            real_counter += 1'''
    train_loss = train(model, train_loader, device, optimizer, crit)
    print('Sythetic sample from' +str(i)+' epoch', synthetic_counter)
    print('real sample from ' +str(i)+' epoch', synthetic_counter)
    loss_values.append(train_loss)

    train_metrics = evaluate(model, train_loader, device, computed_loss=train_loss)
    val_metrics = evaluate(model, val_loader, device, loss_crit=crit)

    metrics_train.append(train_metrics)
    metrics_val.append(val_metrics)

    #if num_epochs <= 25:
    print_evaluation_train_val(epoch, train_metrics, val_metrics)
    metrics_df_train = pd.DataFrame(metrics_train)
    metrics_df_train.to_csv('train_metrics_info32.csv')

    metrics_df_val = pd.DataFrame(metrics_val)
    metrics_df_val.to_csv('val_metrics_info32.csv')
    torch.save(model.state_dict(), 'combined_model_info32')
plot_loss(num_epochs, loss_values, figsize=10, textsize=15)
plot_classification_metrics_train_val(num_epochs, metrics_train, metrics_val, figsize=10, textsize=15)