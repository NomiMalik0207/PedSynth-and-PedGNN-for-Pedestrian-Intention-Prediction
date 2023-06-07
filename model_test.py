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
from SkeletonDataset_alphapose import * #for JAAD
from SkeletonsDataset import * #for Carla
from ModelTrainEvaluate import *
#from MetricsPlots import *
from SkeletonDataset_alphapose import *
import time
import torch



info=64

# First element of training subset:


# Number of nodes:
subset='val'#Only for JAAD
datasetName = "JAAD" #when using JAAD
embed_dim=3 #3 for JAAD+alphapose
numberOfNodes=17
numberOfClasses=2
batch_size=1000
device = torch.device('cpu')
#torch.cuda.set_device(1)
model = SpatialTemporalGNN(embed_dim, numberOfClasses, numberOfNodes, net='GConvGRU', filterSize=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
crit = torch.nn.BCELoss()#weight=class_weights)
modelFileName = '/home/nriaz/PycharmProjects/abel/Code/PIE_dataset/combined_trained_models/model-PedGraphData4frames/train_combine_model189'

test_dataset = SkeletonsDataset('/media/nriaz/NaveedData/PIE_dataset/train_val_test/new_split_using_own/test_pie_17.csv', normalization='minmax',
                                 target='crossing', info=info,
                                 numberOfJoints=17, remove_undetected=False)
#test_dataset = Skeletons_Dataset('/home/nriaz/PycharmProjects/abel/Code/Abel/jaad_ped_graph/Jaad_323/jaad_default_test.csv',numberOfJoints=17, normalization='minmax',
                               #target='cross', info=info)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model_path = '/home/nriaz/PycharmProjects/abel/Code/PIE_dataset/combined_trained_models/model-PedGraphData4frames/train_combine_model189'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

start_time = time.time()
test_metrics = evaluate(model, test_loader, device)
end_time = time.time()
test_time_epoch = (end_time - start_time) / 60

test_metrics["Test time (minutes)"] = test_time_epoch

print_evaluation_test(test_metrics)

metrics_df_test = pd.DataFrame([test_metrics])
metrics_df_test.to_csv(modelFileName + 'test_metrics.csv')