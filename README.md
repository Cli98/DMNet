# Density map guided object detection in aerial image

## Motivation

Aerial image detection problem is challenging to solve for its unique properties. More objects are small (even something looks large on the ground), more pixels are background and more scale variance within images. Aside from that, data may be imbalanced in the real word. Not to say occlusions and truncations for some dense conditions.

In this paper, a novel method (DMNet) is proposed to enhance the performance of general detectors on aerial image dataset. The observation for the skewed distribution between foreground and background inspires our method, which leads to cropping strategy. By applying this, we want to remove as many background pixels as possible, which makes task easier.

After reviewing recent literature, we find density map, that is commonly shown in dense object counting(such as face counting), can aid our research. Density map has the ability to simplify detection problem to more unify manner. Instead of analyze the distribution/coordination/categories for bounding boxes, why not to count their occurrence (pixels per cell), as we only need to crop them out as much as possible? Then we can define a threshold and filter out background in one shot.

That's how DMNet works. To further clarify, please check the following plots for the structure of DMNet. 

<p align="center">
    <img width=620 height=210 src="Images/Figure 1.png"/>
</p>

If you are interested to see more details, please feel free to check the [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf) for more details.

## Demo

Here we provide one video demo for DMNet. The video comes from Visiondrone 2018 dataset, which is a typical one for aerial image detection. Feel free to have a check!

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

The pretrain weight can be accessed [here](https://drive.google.com/file/d/1tpO_58NLNIPXhOYnnaiifuqqLoLZT2i9/view?usp=sharing).

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
