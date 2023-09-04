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
from SkeletonsDataset import *
from SkeletonDataset_alphapose import *
from ModelTrainEvaluate import *
from MetricsPlots import *

import time
import torch

info = 8

train_dataset = SkeletonsDataset('/path/to/dataset/csv/',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)
val_dataset = SkeletonsDataset('/path/to/dataset/csv/',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)




train_dataset.loadedData[['video','frame','crossing']]

numberOfClasses = 2
import pdb
#pdb.set_trace()
y = train_dataset.loadedData['crossing'].to_numpy()
y = np.where(y==1, 1, 0)
bc = np.bincount(y)

class_weights = len(train_dataset.loadedData) / (numberOfClasses * bc)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print('class_weights:', class_weights)
train_dataset.shuffle()
t0 = train_dataset[0]

# Node features:
t1 = t0.x_temporal[0]

# Number of nodes:
numberOfNodes = t1.shape[0]

# Number of dimensions of each node features:
embed_dim = t1.shape[1]

print('Number of nodes per skeleton:', numberOfNodes)
print('Number of features per node:', embed_dim)

#Training Loop
num_epochs = 60
batch_size = 1000

device = torch.device('cuda')
torch.cuda.set_device(0)
model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)
crit = torch.nn.BCELoss()#weight=class_weights)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

loss_values = []

metrics_train = []
metrics_val = []

start_time_total_training = time.time()
for epoch in range(num_epochs):

    start_time = time.time()
    train_loss = train(model, train_loader, device, optimizer, crit)
    end_time = time.time()
    train_time_epoch = (end_time - start_time) / 60
    loss_values.append(train_loss)

    train_metrics = evaluate(model, train_loader, device, computed_loss=train_loss)

    start_time = time.time()
    val_metrics = evaluate(model, val_loader, device, loss_crit=crit)
    end_time = time.time()
    val_time_epoch = (end_time - start_time) / 60

    train_metrics["Epoch"] = int(epoch)
    val_metrics["Epoch"] = int(epoch)

    train_metrics["Step time (minutes)"] = train_time_epoch
    val_metrics["Time (minutes)"] = val_time_epoch

    metrics_train.append(train_metrics)
    metrics_val.append(val_metrics)

    if num_epochs <= 25 or epoch % 10 == 0:
        print_evaluation_train_val(epoch, train_metrics, val_metrics)

    modelFileName = '/path/to/save/model/' + str(info)
    torch.save(model.state_dict(), modelFileName + 'Epoch_' + str(epoch))

    metrics_df_train = pd.DataFrame(metrics_train)
    metrics_df_train.to_csv(modelFileName + 'train_metrics.csv')

    metrics_df_val = pd.DataFrame(metrics_val)
    metrics_df_val.to_csv(modelFileName + 'val_metrics.csv')

end_time_total_training = time.time()
time_total_training = (end_time_total_training - start_time_total_training) / 60

if time_total_training < 60:
    print("Total training time during {:d} epochs: {:0.2f} minutes.".format(num_epochs, time_total_training))
else:
    print("Total training time during {:d} epochs: {:0.2f} hours.".format(num_epochs, time_total_training / 60))
plot_loss(num_epochs, loss_values, figsize=10, textsize=15)
plot_classification_metrics_train_val(num_epochs, metrics_train, metrics_val, figsize=10, textsize=15)
