# SHS_NET joint detection phase
The joint detection phase aims to identify the joint coordinates in the image by weakly supervised learning using heat-mapped images based on U-Net. An overview of this phase follows.

![joint_detection2](https://user-images.githubusercontent.com/80377824/171571670-1247f528-e6bf-451d-98e1-0bbd45195f6c.png)


# Code Architecture
<pre>
.　　
├── data_set                               # hand radiographs for training.   
│   ├──images                              # Directory to store hand radiographs generated by orientation model. Please copy the images generated in the output directory of orientation_phase to this directory.
│   ├──masks                               # Directory to store heatmaps generated by heatmap_generator.ipynb
│   │  ├──0                                # Directory where the heatmap image of joint number 0 (IP joint in this model) is stored
│   │  ├──1                                # Directory where the heatmap image of joint number 1 (PIP2 joint in this model) is stored
│   │    ...
│   ├──test                                # Directory to store hand radiographs for testing and each joint image that was cropped 
│   └──coord_list.csv                      # Correct label list of 15 joint coordinates for each hand radiograph. This list is used in heatmap_generator.ipynb to generate a heatmap from the joint coordinates                    
├── models                      
│   └── model name                         # Directory to store config, log and weight parameter files               
├── README.md                              # README file   
├── U-Net_base_joint_coord_detector.ipynb  # main code of joint detection phase
├── adabound.py                            # code for adabound, a type of optimizer
├── archs.py                               # code for architecuture of U-Net  
├── dataset.py                             # code for making data-loader from images in hand_all_rotation dir and image_list_hand_ver3.csv  
├── heatmap_generator.ipynb                # code for generating heatmaps
├── losses.py                              # code for loss function  
├── metrics.py                             # code for calcurating SDR(Standard dimension ratio)
├── train_val.py                           # code for training and validation  
└── utils.py                               # common useful modules (to make scheduler, optimizer, label maker for training and validation etc.)  
</pre> 


# References
- We generated heatmap images which encode the “pseudo-probability” of joint coordinate being at a particular pixel location. This method is also used for cell segmentation, and we referred to it as well. You can find it [here](https://github.com/naivete5656/WSISPDR)  
Nishimura K, Ker DFE, Bise R. Weakly Supervised Cell Instance Segmentation by Propagating from Detection Response. Published Online First: 29 November 2019. doi:https://doi.org/10.48550/arXiv.1911.13077  
Copyright (c) 2019 Kazuya Nishimura  
The code of WSISPDR is released under the MIT License.  
[https://github.com/naivete5656/WSISPDR/blob/master/LICENSE](https://github.com/naivete5656/WSISPDR/blob/master/LICENSE)

- THe original paper of U-Net is below.  
Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. Published Online First: 18 May 2015.http://arxiv.org/abs/1505.04597  
The source code for U-Net can be found [here](https://github.com/4uiiurz1/pytorch-nested-unet)  
Copyright (c) 2018 Takato Kimura  
The source code for U-Net is released under the MIT License  
[https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/LICENSE](https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/LICENSE)
