# Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction
By Muhammad Naveed Riaz, Maciej Wielgosz, Abel García Romera, Antonio M. López 

**Abstract:**Pedestrian intention prediction is crucial for autonomous driving. In particular, knowing if pedestrians are going to cross in front of the ego-vehicle is core to performing safe and comfortable maneuvers. Creating accurate and fast models that predict such intentions from sequential images is challenging. A factor contributing to this is the lack of datasets with diverse crossing and non-crossing (C/NC) scenarios. We address this scarceness by introducing a framework, named ARCANE, which allows programmatically generating synthetic datasets consisting of C/NC video clip samples. As an example, we use ARCANE to generate a large and diverse dataset named PedSynth. We will show how PedSynth complements widely used real-world datasets such as JAAD and PIE, so enabling more accurate models for C/NC prediction. Considering the onboard deployment of C/NC prediction models, we also propose a deep model named PedGNN, which is fast and has a very low memory footprint. PedGNN is based on a GNN-GRU architecture that takes a sequence of pedestrian skeletons as input to predict crossing intentions. 

## Introduction
This is official repo of our article titles as 'Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction' accepted in ITSC 2023. This repo proves that sythtic dataset along with real world dataset can boost the performance of models for Pedestrian Intention Prediction.

 

## Spatial-Temporal Graph Convolutional Network
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

## PedSynth dataset
To regenrate the synthetic dataset like PedSynth, you can find all the information about ARCANE and PedSynth [here](https://github.com/wielgosz-info/carla-pedestrians/blob/main/README.md). For theoratical insights about ARCANE and PedSynth, you can find technical report [here](https://arxiv.org/abs/2305.00204). You can also download [PedSynth](https://project-arcane.eu/datasets/basic-pedestrians-crossing/) used in our experiments.
## Requirements
Our experiments used following settings
* python 3.10.6
* pytorch 1.10 with cuda 10.2
* pytorch geometric. 

or run `environment.yml` to have all the requirements.

## Data preprocessing
First of all, we extracted all the pedestrian skeletons from the video frames of ([JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)/[PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)) using bounding box values and their respective ground-truth of C/NC. Because the annotatrion files of JAAD and PIE are in .xml format. We change it to a single .csv file. To do so, please use `data_preprocessing/jaad_xml_to_csv.py, data_preprocessing/pie_xml_to_csv.py`. 
## Training
To train our PedGNN, first of all one need to extract the pose coordinated of pedestrians inside the frame and its respective labels of C/NC in the form of `.cvs` file. After that you can run training code. `SkeletonsDataset.py` will preprocess the dataset according to our PedGNN input. `GNN.py` have model structure. Finally, training can be started using `train_pedsynth.py` and other to train PedGNN on single dataset. For combine training settings, please refer to `combine_training.py` file. 
Remember to change the path of your dataset in training file.

## Testing
To test the PedGNN model on testing set of any dataset, run `model_test.py` file. 

## Citation
If you are using our work, please cite
```
@inproceedings{Riaz2023PedSynth,
    author       = {Riaz, Muhammad Naveed, and Wielgosz, Maciej and Romera, Abel Garc{\'i}a, and L{\'o}pez, Antonio M},
    title        = {Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction},
    booktitle    = {2023 IEEE Intelligent Transportation Systems Conference (ITSC)}
    year         = {2023}
}
```

```
@Misc{wielgosz2023carla,
    author       = {Wielgosz, Maciej and L{\'o}pez, Antonio M and Riaz, Muhammad Naveed},
    title        = {{CARLA-BSP}: a simulated dataset with pedestrians},
    howpublished = {arXiv:2305.00204},
    year         = {2023}
}
```
## Funding

This research has been supported by the Spanish Grant Ref. PID2020-115734RB-C21 funded by MCIN/AEI/10.13039/50110001103

<img src="MICINN_Gob_AEI_1.jpg" width="300" />

## Acknowledgements

Antonio M. López acknowledges the financial support to his general research activities given by ICREA under the ICREA Academia Program. Muhammad Naveed Riaz acknowledges the financial support to perform his Ph.D. given by the grant 2021 FI SDUR/00281. The authors acknowledge the support of the Generalitat de Catalunya CERCA Program and its ACCIO agency to CVC’s general activities.



