# SHS_NET orientation_phase
The orientation phase aims to align the orientation of the hand radiographs and furthermore, in the case of a two-handed image, to split the image into one hand. An overview of this phase follows.

![orientation_phase](https://user-images.githubusercontent.com/80377824/171569785-73d6740e-beea-4c92-b4e5-82ae4fe24da6.png)

# Code Architecture
<pre>
.　　
├── hand_all_rotation           # hand radiographs for training.   
├── hand_test                   # hand radiographs for testing (align the orientation and/or split)                    
├── orientation_pred_log       
│   └── models                  # Directory to store config, log and weight parameter files  
├── output                      # Directory to store output images from the model  
├── Orientation_detector.ipynb  # main code of orientation phase  
├── README.md                   # README file  
├── adabound.py                 # code for adabound, a type of optimizer  
├── archs.py                    # code for architecuture of EfficientNet b0  
├── dataset.py                  # code for making data-loader from images in hand_all_rotation dir and image_list_hand_ver3.csv  
├── image_list_hand_ver3.csv    # csv file including image id and true label  
├── losses.py                   # code for loss function  
├── train_val.py                # code for training and validation  
└── utils.py                    # common useful modules (to make scheduler, optimizer, label maker for training and validation etc.)  
</pre> 

This repository contains dummy images obtained from [RSNA-Pediatric-Bone-Age-Challenge-2017](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017)   
Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. The RSNA Pediatric Bone Age Machine Learning Challenge. Radiology 2018; 290(2):498-503.

code of adabound can be found [here](https://github.com/Luolc/AdaBound)  
Luo L, Xiong Y, Liu Y, et al. Adaptive Gradient Methods with Dynamic Bound of Learning Rate. Published Online First: 26 February 2019.http://arxiv.org/abs/1902.09843
