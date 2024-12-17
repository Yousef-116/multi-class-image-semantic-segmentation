# Instructions:
1.	Clone repository.
2.	Install required libraries (pip install -r requirements.txt).
3.	Download dataset (IPProjectDataset24).
4.	Extract dataset (rarfile library).
5.	Run python train.py for training.
6.	Run python predict.py for predictions.
Code Documentation
MarkDown
# train.py
# Train U-Net, FCN, and SegNet models

# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Model
...

# Define models
def GiveMeUnet(inputImage, numFilters=16, droupouts=0.1, doBatchNorm=True):
...

def SimpleFCN(inputImage, numFilters=32):
...

def GiveMeSegNet(inputImage, numFilters=16, droupouts=0.1, doBatchNorm=True):
...

# Train models
retVal = myTransformer.fit(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), epochs=85, ...)
MarkDown
# predict.py
# Make predictions using trained models

# Import libraries
import numpy as np
import cv2
...

# Load trained models
unet_model = tf.keras.models.load_model('unet_model.h5')
fcn_model = tf.keras.models.load_model('fcn_model.h5')
segnet_model = tf.keras.models.load_model('segnet_model.h5')

# Make predictions
sixteenPrediction, actuals, masks = predict16(framObjValidation, fcn_model)

