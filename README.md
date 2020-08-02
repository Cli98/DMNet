# Visual Explanation for Deep Metric Learning

This work explores the visual explanation for deep metric learning and its applications. We show that the proposed framework can be directly deployed to a large range of metric learning applications and provides valuable information for understanding the model. See the arxiv paper for details.

We provide runnable demos for activation decomposition and Grad-CAM (with our variants) on Image Retrieval, Person Re-identification, Face Verification, and Geo-localization. If you find this repository useful in your project, please consider citing:

	@article{zhu2019visual,
	  title={Visual Explanation for Deep Metric Learning},
	  author={Zhu, Sijie and Yang, Taojiannan and Chen, Chen},
	  journal={arXiv preprint arXiv:1909.12977},
	  year={2019}
	}

## Requirement
	- Python >= 3.5, Opencv, Numpy, Matplotlib
	- PyTorch >= 1.0
	- Tensorflow == 1.13.1 only for Geo-localization

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
