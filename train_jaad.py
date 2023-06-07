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
from SkeletonDataset_alphapose import *
from ModelTrainEvaluate import *
from ped_graph23 import *
from MetricsPlots import *
import torch

info = 32

train_dataset = Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/jaad_ped_graph/Jaad_323/jaad_default_train.csv',numberOfJoints=17,
                                 normalization='minmax', target='cross', info=info)
#train_dataset.loadedData[['video','frame','decision_point','keypoints','crossing']]

val_dataset = Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/jaad_ped_graph/Jaad_323/jaad_default_val.csv',numberOfJoints=17, normalization='minmax',
                               target='cross', info=info)



#val_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/Abel/JAAD/val_annotations_with_skeletons_maciek.csv', normalization='minmax',
                               #target='cross', info=info)

val_dataset.shuffle()

#Training
numberOfClasses = 2

y = train_dataset.loadedData['crossing'].to_numpy()
y = np.where(y==1, 1, 0)
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
batch_size = 500

device = torch.device('cpu')
#model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)
model = pedMondel(False, False, False, False, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()  # weight=class_weights)

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
torch.save(model.state_dict(), '/home/nriaz/PycharmProjects/abel/Code/JaaD_trained_models/Approach_2-3_1f')