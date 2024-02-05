# Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction
By Muhammad Naveed Riaz, Maciej Wielgosz, Abel García Romera, Antonio M. López 

**Abstract:**Pedestrian intention prediction is crucial for autonomous driving. In particular, knowing if pedestrians are going to cross in front of the ego-vehicle is core to performing safe and comfortable maneuvers. Creating accurate and fast models that predict such intentions from sequential images is challenging. A factor contributing to this is the lack of datasets with diverse crossing and non-crossing (C/NC) scenarios. We address this scarceness by introducing a framework, named ARCANE, which allows programmatically generating synthetic datasets consisting of C/NC video clip samples. As an example, we use ARCANE to generate a large and diverse dataset named PedSynth. We will show how PedSynth complements widely used real-world datasets such as JAAD and PIE, so enabling more accurate models for C/NC prediction. Considering the onboard deployment of C/NC prediction models, we also propose a deep model named PedGNN, which is fast and has a very low memory footprint. PedGNN is based on a GNN-GRU architecture that takes a sequence of pedestrian skeletons as input to predict crossing intentions. 

## Introduction
Welcome to the official repository for our article, "Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction," which was accepted in ITSC 2023. Our research demonstrates that incorporating a synthetic dataset alongside real-world data can significantly enhance the performance of models for Pedestrian Intention Prediction.


## Spatial-Temporal Graph Convolutional Network
Our proposed Spatial-Temporal model is structured to predict whether a pedestrian will cross the street in a future frame. This is based on analyzing a set of *n* frames and their corresponding *n* skeletons, using a sliding window to group the movements and trajectory of the pedestrian. The goal is to accurately determine if the pedestrian will act as a crossing or not in the near future.
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
If you want to create a synthetic dataset similar to PedSynth, you can access all the necessary information about ARCANE and PedSynth [here](https://github.com/wielgosz-info/carla-pedestrians/blob/main/README.md) . To gain a deeper understanding of ARCANE and PedSynth, you can refer to the technical report available [here](https://arxiv.org/abs/2305.00204). Additionally, you can download the PedSynth used in our experiments from [here](http://datasets.cvc.uab.es/PedSynth/wide_camera_pedestrians.tar.gz).

As an example, here is trimmed PedSynth clips PedSynth as visualizations.
![PedSynth_samples2.mp4](results/PedSynth_samples2.gif)
## Requirements
Our experiments used following settings
* python 3.10.6
* pytorch 1.10 with cuda 10.2
* pytorch geometric. 

or run `environment.yml` to have all the requirements.

## Data preprocessing
To begin with, we utilized bounding box values and the corresponding C/NC ground-truth to extract all pedestrian skeletons from the video frames of [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/). Since the annotation files for both are in .xml format, we converted them into a single .csv file using `data_preprocessing/jaad_xml_to_csv.py` and `data_preprocessing/pie_xml_to_csv.py` scripts. 
## Training
The first step to train our PedGNN is to extract the pose coordinates of pedestrians in the frame, along with their respective C/NC labels, in the form of a `.cvs` file. Once this is done, the training code can be run. The `SkeletonsDataset.py` file pre-processes the dataset according to PedGNN's input needs, while the `GNN.py` file contains the model structure. To train PedGNN on a single dataset, use `train_pedsynth.py` and others. For combined training settings, refer to the `combine_training.py` file. Remember to change the path of your dataset in the training file.

## Testing
To test the PedGNN model on testing set of any dataset, run `model_test.py` file.

## Model Visualizations
#Videos from JAAD

<table>
  <tr>
    <td><img src="results/video_02171-1.gif" alt="Video 1"></td>
    <td><img src="results/video_0245-1.gif" alt="Video 2"></td>
  </tr>
</table>



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



