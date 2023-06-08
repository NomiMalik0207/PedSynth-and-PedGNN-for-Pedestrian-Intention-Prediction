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
#from SkeletonDataset_alphapose import *
from SkeletonsDataset import *
from ModelTrainEvaluate import *
from MetricsPlots import *
from torchvision import transforms
import torch
#from ped_graph23 import pedMondel
import numpy as np
import os

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


info = 32
import numpy as np


train_dataset = SkeletonsDataset('/path/to/dataset/csv/',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)

val_dataset = SkeletonsDataset('/path/to/dataset/csv/',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)

train_dataset.loadedData[['video','frame','crossing']]
totalRows = len(train_dataset.loadedData)
crossingRows = len(train_dataset.loadedData[train_dataset.loadedData['crossing']==True])
nocrossingRows = len(train_dataset.loadedData[train_dataset.loadedData['crossing']!=True])

print('Training dataset total rows:', totalRows)
print('Training dataset crossing class samples:', crossingRows)
print('Training dataset not-crossing class samples:', nocrossingRows)

val_dataset.loadedData[['video','frame','crossing']]
totalRows = len(val_dataset.loadedData)
crossingRows = len(val_dataset.loadedData[val_dataset.loadedData['crossing']==True])
nocrossingRows = len(val_dataset.loadedData[val_dataset.loadedData['crossing']!=True])

print('validation dataset total rows:', totalRows)
print('validation dataset crossing class samples:', crossingRows)
print('validation dataset not-crossing class samples:', nocrossingRows)

numberOfClasses = 2

y = train_dataset.loadedData['crossing'].to_numpy()
y = np.where(y==1, 1, 0)#if condition y==1 is true, funtion will return 1, otherwise 0.
bc = np.bincount(y)

class_weights = len(train_dataset.loadedData) / (numberOfClasses * bc)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print('class_weights:', class_weights)
train_dataset.shuffle()
# First element of training subset:
t0 = train_dataset[0]

# Node features:
t1 = t0.x_temporal[0]

# Number of nodes:
numberOfNodes = t1.shape[0]

# Number of dimensions of each node features:
embed_dim = t1.shape[1]

print('Number of nodes per skeleton:', numberOfNodes)
print('Number of features per node:', embed_dim)
num_epochs = 100
batch_size =200

device = torch.device('cuda')
model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002,weight_decay=1e-3)#sometime if the learning rate is too small(0.0005) in case of PIE is not working and giving 
class_weights = class_weights.to(device)
crit= torch.nn.BCEWithLogitsLoss(weight=class_weights)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

loss_values = []
metrics_train = []
metrics_val = []

for epoch in range(num_epochs):

    train_loss = train(model, train_loader, device, optimizer, crit)
    loss_values.append(train_loss)

    train_metrics = evaluate(model, train_loader, device, computed_loss=train_loss)
    val_metrics = evaluate(model, val_loader, device, loss_crit=crit)

    metrics_train.append(train_metrics)
    metrics_val.append(val_metrics)

    #if num_epochs <= 25:
    print_evaluation_train_val(epoch, train_metrics, val_metrics)
    metrics_df_train = pd.DataFrame(metrics_train)
    modelFileName = 'path/to/save/weights/'+str(info)+'frames/'
    isExist = os.path.exists(modelFileName)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(modelFileName)
        #print("The new directory is created!")
    metrics_df_train.to_csv(modelFileName+'train_metrics.csv')
    metrics_df_val = pd.DataFrame(metrics_val)
    metrics_df_val.to_csv(modelFileName+'val_metrics.csv')
    torch.save(model.state_dict(), modelFileName+'train_model_ped_graph_pie'+str(epoch))
plot_loss(num_epochs, loss_values, figsize=10, textsize=15)
plot_classification_metrics_train_val(num_epochs, metrics_train, metrics_val, figsize=10, textsize=15)
