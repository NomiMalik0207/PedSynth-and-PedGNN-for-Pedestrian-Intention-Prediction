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

# Define the transformations
def random_rotation(keypoints, max_angle=30):
    # Randomly rotate the keypoints
    angle = np.random.uniform(-max_angle, max_angle)
    rad = np.deg2rad(angle)
    cos_rad = np.cos(rad)
    sin_rad = np.sin(rad)
    rot_matrix = np.array([[cos_rad, -sin_rad], [sin_rad, cos_rad]])
    rotated_keypoints = np.dot(rot_matrix, keypoints.T).T
    return rotated_keypoints

def random_scaling(keypoints, max_scale=0.2):
    # Randomly scale the keypoints
    scale = np.random.uniform(1-max_scale, 1+max_scale)
    scaled_keypoints = keypoints * scale
    return scaled_keypoints

def random_flipping(keypoints, p=0.5):
    # Randomly flip the keypoints
    if np.random.uniform() < p:
        left = [0, 5, 6, 7, 8, 9]
        right = [1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16]
        flipped_keypoints = np.copy(keypoints)
        flipped_keypoints[left, 0] = keypoints[right, 0]
        flipped_keypoints[right, 0] = keypoints[left, 0]
        return flipped_keypoints
    else:
        return keypoints

# Define a function to apply the transforms to the data
def apply_transforms(keypoints):
    # Apply the transforms to the keypoints
    transformed_keypoints = random_rotation(keypoints)
    transformed_keypoints = random_scaling(transformed_keypoints)
    transformed_keypoints = random_flipping(transformed_keypoints)
    return transformed_keypoints


#torch.manual_seed(42)

'''data_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()
])'''
#transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)

train_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/data/carla_joints_using_alphapose/PIE_PedGraph_data/train_PedGraph_setting_v1.csv',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)
#train_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/PIE_dataset/train_pie.csv', target='crossing', info=info,
                                 #numberOfJoints=19, remove_undetected=False)

val_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/data/carla_joints_using_alphapose/PIE_PedGraph_data/val_PedGraph_setting_v2.csv',
                                 normalization='minmax', target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)
#val_dataset = SkeletonsDataset('/home/nriaz/PycharmProjects/abel/Code/PIE_dataset/val_pie.csv',target='crossing', info=info,
                                 #numberOfJoints=19, remove_undetected=False)

#val_dataset.shuffle()
#train_dataset.loadedData[['video','frame','skeleton','crossing']]

train_dataset.loadedData[['video','frame','crossing']]
#val_dataset.loadedData[['video','frame','skeleton','crossing']]
totalRows = len(train_dataset.loadedData)
crossingRows = len(train_dataset.loadedData[train_dataset.loadedData['crossing']==True])
nocrossingRows = len(train_dataset.loadedData[train_dataset.loadedData['crossing']!=True])

print('Training dataset total rows:', totalRows)
print('Training dataset crossing class samples:', crossingRows)
print('Training dataset not-crossing class samples:', nocrossingRows)

val_dataset.loadedData[['video','frame','crossing']]
#val_dataset.loadedData[['video','frame','skeleton','crossing']]
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
#train_dataset.shuffle()
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
#model = pedMondel(frames= False, velocity=False, seg=False, h3d=False, n_clss=3)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
from torch.optim.lr_scheduler import StepLR
#schduler = StepLR(optimizer, step_size=10, gamma=0.1)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002, weight_decay= 0.001)
#class_weights.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002,weight_decay=1e-3)#sometime if the learning rate is too small(0.0005) in case of PIE is not working and giving error of zero division
#crit = torch.nn.BCELoss(weight=class_weights)
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
    modelFileName = '/home/nriaz/PycharmProjects/abel/Code/PIE_dataset/trained_models_pie/openpos_models/model-pedgraph_data'+str(info)+'frames/'
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