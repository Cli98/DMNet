# Density map guided object detection in aerial image

## Motivation

Aerial image detection problem is challenging to solve for its unique properties. More objects are small (even something looks large on the ground), more pixels are background and more scale variance within images. Aside from that, data may be imbalanced in the real word. Not to say occlusions and truncations for some dense conditions.

In this paper, a novel method (DMNet) is proposed to enhance the performance of general detectors on aerial image dataset. The observation for the skewed distribution between foreground and background inspires our method, which leads to cropping strategy. By applying this, we want to remove as many background pixels as possible, which makes task easier.

After reviewing recent literature, we find density map, that is commonly shown in dense object counting(such as face counting), can aid our research. Density map has the ability to simplify detection problem to more unify manner. Instead of analyze the distribution/coordination/categories for bounding boxes, why not to count their occurrence (pixels per cell), as we only need to crop them out as much as possible? Then we can define a threshold and filter out background in one shot.

That's how DMNet works. To further clarify, please check the following plots for the structure of DMNet. 

<p align="center">
    <img width=620 height=210 src="Images/Figure 1.png"/>
</p>

If you are interested to see more details, please feel free to check the the arxiv paper for more details.

## Demo

Here we provide one video demo for DMNet. The video comes from Visiondrone 2018 dataset, which is a typical one for aerial image detection. Feel free to have a check!

<p align="center">
    <img width=600 height=337 src="Images/demo.gif"/>
</p>

If you find this repository useful in your project, please consider citing:

	@misc{li2020density,
	    title={Density Map Guided Object Detection in Aerial Images},
	    author={Changlin Li and Taojiannan Yang and Sijie Zhu and Chen Chen and Shanyue Guan},
	    year={2020},
	    eprint={2004.05520},
	    archivePrefix={arXiv},
	    primaryClass={cs.CV}
	}

## Requirement
	- Python >= 3.5, Opencv, Numpy, Matplotlib, tqdm
	- PyTorch >= 1.0
	- mmdetection >=2.0

## Image Retrieval
Please download our pretrained model ([Multi_similarity](https://drive.google.com/open?id=1Wigw2bZfuPK5v2FSnqcsisIjVweTnb7l)) and put it in "./Image_Retrieval/Model/". Then run the demo:
	
	python Image_Retrieval/demo.py
<img width=280 height=210 src="Image_Retrieval/Images/Figure_1.png"/><img width=280 height=210 src="Image_Retrieval/Images/Figure_2.png"/><img width=280 height=210 src="Image_Retrieval/Images/Figure_3.png"/><img width=280 height=210 src="Image_Retrieval/Images/Figure_4.png"/><img width=280 height=210 src="Image_Retrieval/Images/Figure_5.png"/>

You may download the [CUB](http://www.vision.caltech.edu/visipedia/CUB-200.html) dataset to generate more results. Enjoy!

## Person Re-identification
Please download our pretrained model ([strong baseline](https://drive.google.com/open?id=1ZYsvJ3g8YuXDg8S7rtLApUOAb4peqDe2)) and put it in "./Person_Re-identification/Model/". Then run the demo:

	python Person_Re-identification/demo.py
<img width=280 height=210 src="Person_Re-identification/Images/Figure_1.png"/><img width=280 height=210 src="Person_Re-identification/Images/Figure_2.png"/><img width=280 height=210 src="Person_Re-identification/Images/Figure_3.png"/><img width=280 height=210 src="Person_Re-identification/Images/Figure_4.png"/><img width=280 height=210 src="Person_Re-identification/Images/Figure_5.png"/>

You may download the [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset for more results. Enjoy!

## Face Verification
Please download our pretrained model ([arcface](https://drive.google.com/open?id=1YADdI8PahhpkiiHqDJmK1Bxz7VYIt_L2)) and put it in "./Face_Verification/Model/". Then run the demo:

	python Face_Verification/demo.py
<img width=280 height=210 src="Face_Verification/Images/Figure_1.png"/><img width=280 height=210 src="Face_Verification/Images/Figure_2.png"/><img width=280 height=210 src="Face_Verification/Images/Figure_3.png"/><img width=280 height=210 src="Face_Verification/Images/Figure_4.png"/><img width=280 height=210 src="Face_Verification/Images/Figure_5.png"/><img width=280 height=210 src="Face_Verification/Images/Figure_6.png"/>

You may download the [LFW](http://vis-www.cs.umass.edu/lfw/) or [FIW](https://web.northeastern.edu/smilelab/fiw/) dataset for more results. Enjoy!

## Geo-localization
Please download our pretrained model package ([Siamese-VGG](https://drive.google.com/open?id=1U8zvR6rfYY5A4TbeXuO8mwNjg0KiDQJz)) and put it in "./Geo-localization/Model/". Use `tar -xvf model.ckpt.tar` to extract. Then run the demo:

	python Geo-localization/demo.py
<img width=280 height=210 src="Geo-localization/Images/Figure_1.png"/><img width=280 height=210 src="Geo-localization/Images/Figure_2.png"/><img width=280 height=210 src="Geo-localization/Images/Figure_3.png"/>

Query pixel moving on the query street view image (left) The changing point-specific activation map on the retrieved aerial view image (right)

<img src="https://github.com/Jeff-Zilence/anonymous/blob/master/grd_show.gif"/><img width=300 height=300 src="https://github.com/Jeff-Zilence/anonymous/blob/master/grd_cam.gif"/>

You may download the [CVUSA](https://github.com/viibridges/crossnet) dataset for more results. Enjoy!
*A cleaner yet stronger model is coming soon!*

## Reference
	- https://github.com/bnu-wangxun/Deep_Metric
	- https://github.com/lulujianjie/person-reid-tiny-baseline
	- https://github.com/foamliu/InsightFace.git
	- https://github.com/david-husx/crossview_localisation.git
