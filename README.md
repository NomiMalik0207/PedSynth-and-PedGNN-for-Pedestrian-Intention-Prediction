# Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction
 This repo proves that sythtic dataset along with real world dataset can boost the performance of models for Pedestrian Intention Prediction
## abstract
Pedestrian intention prediction is crucial for autonomous driving. In particular, knowing if pedestrians are going to cross in front of the ego-vehicle is core to performing safe and comfortable maneuvers. Creating accurate and fast models that predict such intentions from sequential images is challenging. A factor contributing to this is the lack of datasets with diverse crossing and non-crossing (C/NC) scenarios. We address this scarceness by introducing a framework, named ARCANE, which allows programmatically generating synthetic datasets consisting of C/NC video clip samples. As an example, we use ARCANE to generate a large and diverse dataset named PedSynth. We will show how PedSynth complements widely used real-world datasets such as JAAD and PIE, so enabling more accurate models for C/NC prediction. Considering the onboard deployment of C/NC prediction models, we also propose a deep model named PedGNN, which is fast and has a very low memory footprint. PedGNN is based on a GNN-GRU architecture that takes a sequence of pedestrian skeletons as input to predict crossing intentions.  
##Spatial-Temporal Graph Convolutional Network

The structure of our proposed Spatial-Temporal model. For a set of *n* frames with their *n* skeletons, predict if in a future frame the pedestrian will perform the action of crossing or not. That is, from the movements and trajectory of the pedestrian during *n* frames (grouped using a sliding window), predict whether he or she will cross the street or not in the near future.

The layers of the model are detailed in the following table:

|     Layer                     |     Input shape    |     Output shape    |
|:------------------------------|:-------------------|:--------------------|
|     Recurrent part:           |     -              |     -               |
|      \|--> GConvGRU           |     [-1, 26, 3]    |     [-1, 26, 3]     |
|      \|--> Dropout   (0.5)    |     [-1, 26, 3]    |     [-1, 26, 3]     |
|      \|--> ReLU               |     [-1, 26, 3]    |     [-1, 26, 3]     |
|     End of recurrent part     |     -              |     -               |
|     Reshape                   |     [-1, 26, 3]    |     [-1, 78]        |
|     Linear                    |     [-1, 78]       |     [-1, 39]        |
|     Dropout (0.5)             |     [-1, 39]       |     [-1, 39]        |
|     ReLU                      |     [-1, 39]       |     [-1, 39]        |
|     Linear                    |     [-1, 39]       |     [-1, 19]        |
|     Dropout (0.5)             |     [-1, 19]       |     [-1, 19]        |
|     ReLU                      |     [-1, 19]       |     [-1, 19]        |
|     Linear                    |     [-1, 19]       |     [-1, 2]         |
|     Softmax                   |     [-1, 2]        |     [-1, 2]         |

The input to the network contains, for each frame, 26*3 elements (for PedSynth) because in CARLA there are 26 different joints, and we input the 3D (x,y, cs) coordinates of them.
