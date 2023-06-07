import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import torch
from torch.utils.data import Dataset
#from torch_geometric.data import Data
from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
#from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class SkeletonsDataset(Dataset):
    
    """
    Allowed values:
    
    csv_path: string
    numberOfJoints: 26, 25 or 18
    normalization: minmax or standardization
    norm_precomputed_values: None or tuple of [max, min] or [mean, std] precomputed values
    target: 'cross' or 'crossing'
    info: 'spatial' or int representing the number of frames for a spatial-temporal approach.
    """
    def __init__(self, csv_path, numberOfJoints=15, normalization='minmax', norm_precomputed_values=None, target='cross', info='spatial', remove_undetected=False, transform=None):
        import pdb
        #pdb.set_trace()
        self.data=[]
        self.csv_path = csv_path
        self.normalization = normalization
        self.norm_precomputed_values = norm_precomputed_values
        self.targetName = target
        self.graphInfo = info
        self.remove_undetected = remove_undetected
        print(numberOfJoints)
        
        if numberOfJoints == 25: # JAAD videos + OpenPose
            #print('jaad+openpose25')
            self.body_parts = {
                "Nose": 0,
                "Neck": 1,
                "RShoulder": 2,
                "RElbow": 3,
                "RWrist": 4,
                "LShoulder": 5,
                "LElbow": 6,
                "LWrist": 7,
                "MidHip": 8,
                "RHip": 9,
                "RKnee": 10,
                "RAnkle": 11,
                "LHip": 12,
                "LKnee": 13,
                "LAnkle": 14,
                "REye": 15,
                "LEye": 16,
                "REar": 17,
                "LEar": 18,
                "LBigToe": 19,
                "LSmallToe": 20,
                "LHeel": 21,
                "RBigToe": 22,
                "RSmallToe": 23,
                "RHeel": 24,
            }
            
            
            self.pose_parts = [
                ["Nose", "Neck"],
                ["Neck", "RShoulder"],
                ["Neck", "LShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "MidHip"],
                ["MidHip", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["RAnkle", "RHeel"],
                ["RHeel", "RBigToe"],
                ["RBigToe", "RSmallToe"],
                ["MidHip", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],
                ["LAnkle", "LHeel"],
                ["LHeel", "LBigToe"],
                ["LBigToe", "LSmallToe"],
                ["Nose", "REye"],
                ["REye", "REar"],
                ["Nose", "LEye"],
                ["LEye", "LEar"],
            ]
            
        elif numberOfJoints == 26: # PedSynth CARLA simulator
            #print('carla26')
            self.body_parts = {
                "crl_root": 0,
                "crl_hips__C": 1,
                "crl_spine__C": 2,
                "crl_spine01__C": 3,
                "crl_shoulder__L": 4,
                "crl_arm__L": 5,
                "crl_foreArm__L": 6,
                "crl_hand__L": 7,
                "crl_neck__C": 8,
                "crl_Head__C": 9,
                "crl_eye__L": 10,
                "crl_eye__R": 11,
                "crl_shoulder__R": 12,
                "crl_arm__R": 13,
                "crl_foreArm__R": 14,
                "crl_hand__R": 15,
                "crl_thigh__R": 16,
                "crl_leg__R": 17,
                "crl_foot__R": 18,
                "crl_toe__R": 19,
                "crl_toeEnd__R": 20,
                "crl_thigh__L": 21,
                "crl_leg__L": 22,
                "crl_foot__L": 23,
                "crl_toe__L": 24,
                "crl_toeEnd__L": 25,
            }
            
            
            self.pose_parts = [
                ['crl_root', 'crl_hips__C'],
                ['crl_hips__C', 'crl_spine__C'],
                ['crl_hips__C', 'crl_thigh__R'],
                ['crl_hips__C', 'crl_thigh__L'],
                ['crl_spine__C', 'crl_spine01__C'],
                ['crl_spine01__C', 'crl_shoulder__L'],
                ['crl_spine01__C', 'crl_neck__C'],
                ['crl_spine01__C', 'crl_shoulder__R'],
                ['crl_shoulder__L', 'crl_arm__L'],
                ['crl_arm__L', 'crl_foreArm__L'],
                ['crl_foreArm__L', 'crl_hand__L'],
                ['crl_neck__C', 'crl_Head__C'],
                ['crl_Head__C', 'crl_eye__L'],
                ['crl_Head__C', 'crl_eye__R'],
                ['crl_shoulder__R', 'crl_arm__R'],
                ['crl_arm__R', 'crl_foreArm__R'],
                ['crl_foreArm__R', 'crl_hand__R'],
                ['crl_thigh__R', 'crl_leg__R'],
                ['crl_leg__R', 'crl_foot__R'],
                ['crl_foot__R', 'crl_toe__R'],
                ['crl_toe__R', 'crl_toeEnd__R'],
                ['crl_thigh__L', 'crl_leg__L'],
                ['crl_leg__L', 'crl_foot__L'],
                ['crl_foot__L', 'crl_toe__L'],
                ['crl_toe__L', 'crl_toeEnd__L']
            ]

        elif numberOfJoints == 17:  # Alphpose+coco simulator
            #print('jaad+alpha17')
            self.body_parts = {
                "Nose": 0,
                "LEye": 1,
                "REye": 2,
                "LEar": 3,
                "REar": 4,
                "LShoulder": 5,
                "RShoulder": 6,
                "LElbow": 7,
                "RElbow": 8,
                "LWrist": 9,
                "RWrist": 10,
                "LHip": 11,
                "RHip": 12,
                "LKnee": 13,
                "RKnee": 14,
                "LAnkle": 15,
                "RAnkle": 16,
            }
         
            #Adding 2 additional joints required when use training on PedSynth CARLA joints and test on alphapose extracted joints
            self.body_parts["Neck"] = (self.body_parts["LShoulder"] + self.body_parts["RShoulder"]) / 2
            self.body_parts["CHip"] = (self.body_parts["LHip"] + self.body_parts["RHip"]) / 2


            self.pose_parts = [
                ["Neck", "Nose"],
                ["Nose", "REye"],
                ["REye", "REar"],
                ["Nose", "LEye"],
                ["LEye", "LEar"],
                ["Neck", "LShoulder"],
                ["Neck", "RShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["Neck", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],

            ]

        elif numberOfJoints == 15:  # Alphpose+coco simulator
            print('carla data with 15 points')
            self.body_parts = {
                "Nose": 0,
                "LEye": 1,
                "REye": 2,
                #"LEar": 3,
                #"REar": 4,
                "LShoulder": 3,
                "RShoulder": 4,
                "LElbow": 5,
                "RElbow": 6,
                "LWrist": 7,
                "RWrist": 8,
                "LHip": 9,
                "RHip": 10,
                "LKnee": 11,
                "RKnee": 12,
                "LAnkle": 13,
                "RAnkle": 14,
            }
         
            self.body_parts["Neck"] = (self.body_parts["LShoulder"] + self.body_parts["RShoulder"]) / 2

            self.pose_parts = [
                ["Neck", "Nose"],
                ["Nose", "REye"],
               # ["REye", "REar"],
                ["Nose", "LEye"],
                #["LEye", "LEar"],
                ["Neck", "LShoulder"],
                ["Neck", "RShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["Neck", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],

            ]

        else:
    
            self.body_parts = {
                "Nose": 0,
                "Neck": 1,
                "RShoulder": 2,
                "RElbow": 3,
                "RWrist": 4,
                "LShoulder": 5,
                "LElbow": 6,
                "LWrist": 7,
                "RHip": 8,
                "RKnee": 9,
                "RAnkle": 10,
                "LHip": 11,
                "LKnee": 12,
                "LAnkle": 13,
                "REye": 14,
                "LEye": 15,
                "REar": 16,
                "LEar": 17,
            }

            self.pose_parts = [
                ["Neck", "RShoulder"],
                ["Neck", "LShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["Neck", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],
                ["Neck", "Nose"],
                ["Nose", "REye"],
                ["REye", "REar"],
                ["Nose", "LEye"],
                ["LEye", "LEar"],
            ]


        self.generateEdges()
        self.generateEdgeWeights()

        self.loadCSV()
        self.processData()


    def __len__(self):
        return len(self.data)

    def loadCSV(self):
        self.loadedData = pd.read_csv(self.csv_path)


    def generateEdges(self):

        directed = False
        import pdb
        #pdb.set_trace()
        numberOfEdges = len(self.pose_parts)
        print('edges are',numberOfEdges)

        if not directed:
            numberOfEdges = numberOfEdges * 2

        # Pytorch COO format:
        # 2 rows and each column is an edge from node_i to node_j
        self.edgeindex = np.zeros(shape=(2, numberOfEdges))

        part = 0
        rev = False
        for col in range(0, numberOfEdges):

            edge = self.pose_parts[part]

            node1 = edge[0]
            node2 = edge[1]

            node1_part = self.body_parts[node1]
            node2_part = self.body_parts[node2]

            self.edgeindex[1 if rev else 0][col] = node1_part
            self.edgeindex[0 if rev else 1][col] = node2_part

            part = part + 1
            if part == len(self.pose_parts):
                part = 0
                rev = True
                
                
    def generateEdgeWeights(self):

        self.edge_weights = np.ones(self.edgeindex.shape[1]) # All edges weight 1
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float)


    def processData(self):

        if self.remove_undetected:
            # Drop all samples in which the skeleton was not detected on the previous pipeline step
            self.loadedData = self.loadedData[self.loadedData['skeleton_detected']==True]

        # Preprocess the target of the network:
        
        crossing = self.loadedData[self.targetName].to_numpy()
        
        if self.targetName == 'cross':
            target_nocross = np.where(crossing=='crossing', 0, 1)
            target_cross = np.where(crossing=='crossing', 1, 0)
        else:
            #if self.targetName == 'crossing':
                #self.loadedData = self.loadedData[self.loadedData['crossing']!=-1]
            
            target_nocross = np.where(crossing==1, 0, 1)
            target_cross = np.where(crossing==1, 1, 0)
        

        target = np.stack([target_nocross, target_cross], axis=1)
        
        
        # Preprocess the input of the network:
        
        skeletonData = self.loadedData['keypoints'].tolist() #skeleton for original 26 keypoints, skeleton_jaad for jaad 15 keypoints
        videoIDs = self.loadedData['video'].tolist()

        y_values = []
        x_values = []
        label_values = []
        
        self.original_skeletons = {} # Skeletons without any normalization or processing for image display

        # For each skeleton of the dataset
        for i, skeleton in enumerate(skeletonData):
            
            skeleton = skeleton.split('], ')

            jointCoords = []

            # Each skeleton is stored as a string representing the list of lists of each joint coordinates
            firstJoint = True
            for joint in skeleton[:-1]:
                coords = joint[2 if firstJoint else 1:]
                firstJoint = False
                        
                coords = np.fromstring(coords, dtype=np.float32, sep=',')

                jointCoords.append(coords)


            lastJoint = np.fromstring(skeleton[-1][1:-2], dtype=np.float32, sep=',')
            jointCoords.append(lastJoint)

            jointCoords = np.asarray(jointCoords)
            
            videoID = videoIDs[i]
            if videoID in self.original_skeletons:
                self.original_skeletons[videoID].append(np.copy(jointCoords))
            else:
                self.original_skeletons[videoID] = [np.copy(jointCoords)]
            
            if self.normalization == 'minmax':
                
                xmax = np.amax(jointCoords, axis=0, keepdims=True)
                
                # Avoid using missing joints in the minimum operation:
                fzeros = np.zeros(shape=(jointCoords.shape[1]))
                jointM = np.where(jointCoords==fzeros, xmax, jointCoords)
                xmin = np.amin(jointM, axis=0, keepdims=True)
                            
                jointCoords_n = (jointCoords - xmin) / ((xmax - xmin) + 1e-20)
                
                # Set the missing joints as [0, 0, ...] also in the final graph array:
                jointCoords = np.where(jointCoords==fzeros, fzeros, jointCoords_n)


            y = np.asarray(target[i])

            x = torch.tensor(jointCoords, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            label = torch.tensor(np.asarray(target_cross[i]), dtype=torch.float)

            y_values.append(y)
            x_values.append(x)
            label_values.append(label)


        x_values = torch.stack(x_values) # Dim0: n samples, Dim1: m nodes per sample, Dim2: k features per node
        
        
        if self.normalization == 'standardization':
            
            # Standardization:
            
            if self.norm_precomputed_values is None:
                xmean = torch.mean(x_values, dim=[0,1], keepdim=True)
                xstd = torch.std(x_values, dim=[0,1], keepdim=True)
            else:
                xmean, xstd = self.norm_precomputed_values
            
            x_values = (x_values - xmean) / xstd
        
            self.xmean = xmean
            self.xstd = xstd
            

        elif self.normalization == 'minmax':
            
            # Min-max normalization:
            
            if self.norm_precomputed_values is None:
                xmax = torch.amax(x_values, dim=[1], keepdim=True)                
                xmin = torch.amin(x_values, dim=[1], keepdim=True)
            else:
                xmax, xmin = self.norm_precomputed_values
            
            x_values = (x_values - xmin) / (xmax - xmin)
            
            self.xmax = xmax
            self.xmin = xmin

            
        if self.normalization == 'max':
            
            # Max normalization:
            
            if self.norm_precomputed_values is None:
                xmax = torch.amax(x_values, dim=[1], keepdim=True)                
            else:
                xmax = self.norm_precomputed_values
            
            x_values = x_values / xmax
            
            self.xmax = xmax

        
        edge_index = torch.tensor(self.edgeindex, dtype=torch.long)
        
        # If each graph is independent, only spatial dimension is used:

        if self.graphInfo == 'spatial':
        
            for iy, x in enumerate(x_values):

                """
                Data class represents a homogeneous graph.
                Constructor parameters:
                
                x: Node feature matrix with shape [num_nodes, num_node_features]
                y: Ground-truth labels with arbitrary shape, in this case [2].
                label: Ground-truth labels (positive class: crossing class) with arbitrary shape, in this case [1].
                edge_index: Graph connectivity in COO format with shape [2, num_edges]
                num_nodes: Number of nodes of the graph.
                """
                data_element = Data(x=x, y=y_values[iy], label=label_values[iy], edge_index=edge_index, num_nodes=x.shape[0], videoID=videoIDs[iy])
                self.data.append(data_element)
                
        else: # Group graphs of the same videos to add the temporal dimension, getting spatial-temporal graphs:
                        
            numFrames = self.graphInfo if isinstance(self.graphInfo, int) else 2
            
            for i in range(numFrames*2 - 1, x_values.shape[0]): #for num frames 3 (5, 239197)n
                
                targets_i = target[i - numFrames*2 + 1:i + 1]
                crossing_class_i = targets_i[:, 1]
                                
                # If the pedestrian is crossing in the numFrames frames of the temporal graph, or in the previous numFrames frames
                if 1 in crossing_class_i:
                    label_i = 1
                    y_value_i = [0, 1]
                else:
                    label_i = 0
                    y_value_i = [1, 0]
                
                label_i = torch.tensor(label_i, dtype=torch.float)
                y_value_i = torch.tensor(y_value_i, dtype=torch.float)
                                                
                currentFrame = i - numFrames*2 + 1
                                
                id_video_i = videoIDs[currentFrame]
                                
                x_temp = []
                
                for j in range(0, numFrames):
                    
                    id_video_j = videoIDs[currentFrame+j]
                                        
                    # Ensure that all frames are from the same video:
                    if id_video_i == id_video_j:
                    
                        x_temp.append(x_values[currentFrame+j])
                        
                    # Frame is from another video:
                    else:
                        break
                import pdb
                #pdb.set_trace()
                if len(x_temp) == numFrames:
            
                    """
                    StaticGraphTemporalSignal works similarly to the Data class, but x is called features, and y is called targets.
                    Features and targets are lists of arrays, in which each array is from a different moment in time (temporal dimension).
                    """
                    #data_element = StaticGraphTemporalSignal(features=x_temp, targets=y_temp, label=label_temp, edge_index=edge_index, edge_weight=self.edge_weights)

                    # StaticGraphTemporalSignal is not compatible with DataLoader, so I use Data class here again, but now x is a list of tensors.
                    data_element = Data(x_temporal=x_temp, y=y_value_i, label=label_i, edge_index=edge_index,
                                        num_nodes=x_values[currentFrame].shape[0], edge_weight=self.edge_weights, videoID=id_video_i)

                    self.data.append(data_element)

    def calculate_neck(self, LShoulder, RShoulder): #In alphapose, we dont have neck coordination we will add using this function
        neck_x = (LShoulder[0] + RShoulder[0]) / 2
        neck_y = (LShoulder[1] + RShoulder[1]) / 2
        return (neck_x, neck_y)


    def showSkeleton(self, videoNum=0, frameNum=0, textSize=14, showLegend=True, frameImage=None, normalizedSkeletons=True, 
                     title='Skeleton preview in 2D', show=True, prediction=None, groundtruth=None):
        
        parts = list(self.body_parts.keys())

        node_coords = {}

        fig = plt.figure(figsize=(10,10)) if show else Figure(figsize=(10,10))
            
        ax = fig.add_subplot(1, 1, 1)

        if normalizedSkeletons:
            skeleton = self.data[videoNum].x_temporal[frameNum][:, 0:2].tolist()
        else:
            skeleton = self.original_skeletons[videoNum][frameNum][:, 0:2].tolist()

        for e, sk in enumerate(skeleton):    
            node_coords[parts[e]] = sk

            ax.scatter(sk[0], sk[1], label=parts[e])

        for edge in self.pose_parts:
            e0 = node_coords[edge[0]]
            e1 = node_coords[edge[1]]

            ax.plot([e0[0], e1[0]], [e0[1], e1[1]], color='gray')

            
        if showLegend:
            ax.legend(loc='best', prop={'size': 11})
            
        if frameImage is not None:
            ax.imshow(frameImage)
            
        if prediction is not None:
            
            if groundtruth is not None:
                
                textColor = 'green' if prediction == groundtruth else 'red'
                
            else:
                
                textColor = 'white'
                
            sk_arr = np.asarray(skeleton)
            sk_arr = sk_arr[list(set(np.where(sk_arr != [0.0, 0.0])[0]))] # Removes the missing joints
            
            if sk_arr.shape[0] == 0: # All joints are missing
                
                textCoords = [0, 0]
                
            else:
                
                textCoords = sk_arr[np.argmin(sk_arr[:, 1])]
                
            ax.text(textCoords[0] * 0.95, textCoords[1] * 0.95, prediction, color=textColor, fontsize='medium', backgroundcolor='black')
        
        if show:
            plt.xticks(size=textSize)
            plt.yticks(size=textSize)
            plt.title(title, size=textSize)
            
        else:
            ax.set_title(title, size=textSize)
        
        if show:
            plt.show()
        else:
            return fig
            
            
    def shuffle(self):
        random.shuffle(self.data)
        return self

    def __getitem__(self, items):
        return self.data[items]

