# Density map guided object detection in aerial image

## Introduction

Object detection in high-resolution aerial images is a challenging task because of 1) the large variation in ob- ject size, and 2) non-uniform distribution of objects. A common solution is to divide the large aerial image into small (uniform) crops and then apply object detection on each small crop. In this paper, we investigate the image cropping strategy to address these challenges. Specifically, we propose a Density-Map guided object detection Network (DMNet), which is inspired from the observation that the object density map of an image presents how objects dis- tribute in terms of the pixel intensity of the map. As pixel intensity varies, it is able to tell whether a region has objects or not, which in turn provides guidance for cropping images statistically. DMNet has three key components: a density map generation module, an image cropping module and an object detector. DMNet generates a density map and learns scale information based on density intensities to form cropping regions. Extensive experiments show that DMNet achieves state-of-the-art performance on two popular aerial image datasets, i.e. VisionDrone [30] and UAVDT [4]. 

<p align="center">
    <img width=2142 height=568 src="Images/Figure 1.png"/>
</p>

If you are interested to see more details, please feel free to check the [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf) for more details.

## Demo

Here we provide one video demo for DMNet. The video comes from Visiondrone 2018 dataset, which is a typical one for aerial image object detection. 

<p align="center">
    <img width=600 height=337 src="Images/demo.gif"/>
</p>

If you find this repository useful in your project, please consider citing:

    @InProceedings{Li_2020_CVPR_Workshops,
        author = {Li, Changlin and Yang, Taojiannan and Zhu, Sijie and Chen, Chen and Guan, Shanyue},
        title = {Density Map Guided Object Detection in Aerial Images},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month = {June},
        year = {2020}
    }

## Requirement
	- Python >= 3.5, Opencv, Numpy, Matplotlib, tqdm
	- PyTorch >= 1.0
	- mmdetection >=2.0

## Dessity map generation

There are already state of art algorithms that can achieve satisfying results on density map generation. In DMNet, the density map generation modular uses MCNN to achieve the task. Yet there are many models can beat MCNN in terms of mean absolute error. How good they are turns out to be a further research direction for DMNet.

We introduce code from Ma to train [MCNN](https://github.com/CommissarMa/MCNN-pytorch). Since the code base is available online, we save the trouble to publish it again in DMNet.

The pretrain weight can be accessed [here](https://drive.google.com/file/d/1J--qH8_djZIsX3YUz9IkysWsfxzKXEqI/view?usp=sharing).

## Image cropping

Once you obtained prediction of density map from density map generation modular, collect all of them and place it your dataset to run image cropping modular

The data should be arranged in following structure before you call any function within this script:

dataset(Train/val/test)

--------images

--------dens (short for density map)

--------Annotations (Optional, but not available only when you conduct inference steps)

Sample running command:

python density_slide_window_official.py . HEIGHT_WIDTH THRESHOLD --output_folder Output_FolderName --mode val

## Object detection

After you obtain your density crops, collect crop images and annotations(txt format in Visiondrone 2018) and make them into proper annotations(COCO or VOC format). Then you can select any state of art object detection algorithm to train your model.

For DMNet, MMdetection is selected as the tool for training Faster-Rcnn detectors.

The pretrain weight can be accessed [here](https://drive.google.com/file/d/16Mu_U_znOn8HCQiBvXePjMQJI1Fjjri2/view?usp=sharing).

If you are not familiar with the process to transform txt annotation to VOC/COCO format, please check

1. create_VOC_annotation_official.py

This script helps you Loading txt annotation and transform to VOC format
The resulting images+annotations will be saved to indicated folders

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)

--------images

--------Annotations (Optional, not available only when you conduct inference steps)

Sample command line to run:

python create_VOC_annotation_official.py ./mcnn_0.08_train_data --h_split 2 --w_split 3 --output_folder
FolderName --mode train

2. VOC2coco_official.py

Loading VOC annotation and transform to COCO format

Normally it should be enough to extract your annotation to VOC format,
which is supported by various of object detection framework. However,
there does exist the needs to obtain annotation in COCO format. And
this script can help you.


The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)

--------images

--------Annotations (XML format, Optional, not available only when you conduct inference steps)

Sample command line to run:

python VOC2coco_official.py Folder_Name --mode train

3. fusion_detection_result_official.py

This script conducts Global-local fusion detection. Namely, the script will fuse detections from both original image and density crops. To use this script, please prepare your data in following structure:

dataset(Train/val/test)

-----mode(Train/val/test)

------Global

--------images

--------Annotations (Optional, not available only when you conduct inference steps)

------Density

--------images

--------Annotations (Optional, not available only when you conduct inference steps)

Sample command line to run:

python fusion_detection_result_official.py

## Reference
	- https://github.com/CommissarMa/MCNN-pytorch
	- https://github.com/open-mmlab/mmdetection
