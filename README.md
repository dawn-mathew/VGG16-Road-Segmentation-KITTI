# VGG16-Road-Segmentation-KITTI
## Introduction
This repository provides essential code and the KITTI road segmentation dataset. The basic code implementation is designed for testing and training a neural network to assess performance and accuracy on this dataset.

## Repo Structure
```
├├── Data Folder
│   └── training
│       ├── calib
│       ├── gt_image_2
│       └── image_2
└── Road_Segmentation.ipynb

```
- `gt_image_2` is a subdirectory containing labeled images (ground truth images) for segmentation. These the masks that are used to train the neural network for the segmentation task.
- `image_2` is a subdirectory where the original images are stored, which are used for segmentation training and testing.
- `Road_Segmentation.ipynb` is a Jupyter Notebook containing the code and documentation related to performing road segmentation tasks on the provided dataset.

## Dependencies
Install all of the following libraries. 
```py
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from tqdm import tqdm
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate 
from tensorflow.keras.layers import Input, Add, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
%matplotlib inline
from IPython.display import HTML
from base64 import b64encode
```
We are using a VGG16 model as the backbone and training it further fot our road segmentation use case.
```py
from tensorflow.keras.applications import VGG16
vgg16_model = VGG16()
```
## Results
