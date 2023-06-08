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

train_dataset_carla = SkeletonsDataset('path/to/csv/dataset',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)

train_dataset_jaad=Skeletons_Dataset('path/to/csv/dataset',numberOfJoints=17,
                                 normalization='minmax', target='cross', info=info )
val_dataset = Skeletons_Dataset('path/to/csv/dataset',numberOfJoints=17, normalization='minmax',
                               target='cross', info=info)
                                
combined_dataset_train = ConcatDataset([train_dataset_carla, train_dataset_jaad])


print('Training len:', len(combined_dataset_train))
totalRows_val = len(val_dataset.loadedData)
crossingRows_val = len(val_dataset.loadedData[val_dataset.loadedData['crossing']==True])
nocrossingRows_val = len(val_dataset.loadedData[val_dataset.loadedData['crossing']!=True])
print('validation dataset total rows:', totalRows_val)
print('validation len:', len(val_dataset))
print('val dataset crossing class samples:', crossingRows_val)
print('val dataset not-crossing class samples:', nocrossingRows_val)

val_dataset.shuffle()

#Training
numberOfClasses = 2

t0 =combined_dataset_train[0]

# Node features:
t1 = t0.x_temporal[0]

# Number of nodes:
numberOfNodes = t1.shape[0]

# Number of dimensions of each node features:
embed_dim = t1.shape[1]#embed_dim = 2

print('Number of nodes per skeleton:', numberOfNodes)
print('Number of features per node:', embed_dim)
num_epochs = 60
batch_size = 300

device = torch.device('cuda')
model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
crit = torch.nn.BCELoss() 
weights = [len(train_dataset_carla)/(len(train_dataset_jaad)+len(train_dataset_carla)) if i <len(train_dataset_carla) else len(train_dataset_jaad)/(len(train_dataset_jaad)+len(train_dataset_carla)) for i in range(len(train_dataset_jaad)+len(train_dataset_carla))]
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(combined_dataset_train, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

loss_values = []

metrics_train = []
metrics_val = []
synthetic_counter = 0
real_counter = 0
for i, epoch in enumerate(range(num_epochs)):
    for i, inputs in enumerate(train_loader):
    train_loss = train(model, train_loader, device, optimizer, crit)
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
