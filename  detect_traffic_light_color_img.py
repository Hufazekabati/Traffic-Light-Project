# Project: How to Detect and Classify Traffic Lights
# Author: Addison Sears-Collins
# Date created: January 17, 2021
# Description: This program uses a trained neural network to 
# detect the color of a traffic light in images.
 
import cv2 # Computer vision library
import numpy as np # Scientific computing library
import object_detection # Custom object detection program
from tensorflow import keras # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
 
FILENAME = "test_red.jpg"
 
# Load the Inception V3 model
model_inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299,299,3))
 
# Resize the image
img = cv2.resize(preprocess_input(cv2.imread(FILENAME)), (299, 299))
 
# Generate predictions
out_inception = model_inception.predict(np.array([img]))
 
# Decode the predictions
out_inception = imagenet_utils.decode_predictions(out_inception)
 
print("Prediction for ", FILENAME , ": ", out_inception[0][0][1], out_inception[0][0][2], "%")
 
# Show model summary data
model_inception.summary()
 
# Detect traffic light color in a batch of image files
files = object_detection.get_files('test_images/*.jpg')
 
# Load the SSD neural network that is trained on the COCO data set
model_ssd = object_detection.load_ssd_coco()
 
# Load the trained neural network 
model_traffic_lights_nn = keras.models.load_model("traffic.keras")
 
# Go through all image files, and detect the traffic light color. 
for file in files:
  (img, out, file_name) = object_detection.perform_object_detection(
    model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
  print(file, out)

