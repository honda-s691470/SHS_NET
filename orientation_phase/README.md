# SHS_NET orientation_phase
The orientation phase aims to align the orientation of the hand radiographs and furthermore, in the case of a two-handed image, to split the image into one hand. An overview of this phase follows.

![orientation_phase](https://user-images.githubusercontent.com/80377824/171557850-b7715e42-8447-40d1-9fbf-9a5b77837cdf.png)

# Code Architecture
<pre>
.　　
├── hand_all_rotation           # hand radiographs for training. This repository contains dummy images obtained from {RSNA-Pediatric-Bone-Age-Challenge-2017}  
├── hand_test                   # hand radiographs for testing (align the orientation and/or split)                    
├── orientation_phase       
│   └── orientation_pred_log               
│       └── models              # Directory to store config, log and weight parameter files  
├── output                      # Directory for storing images output from the model  
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
