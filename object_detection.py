# Object Detection functions module

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2 
import glob
 
# Inception V3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Labels
LABEL_PERSON = 1
LABEL_CAR = 3
LABEL_BUS = 6
LABEL_TRUCK = 8
LABEL_TRAFFIC_LIGHT = 10
LABEL_STOP_SIGN = 13
 
# Accepting boxes function (eliminate double boxes)
def accept_box(boxes, box_index, tolerance):
  box = boxes[box_index]
 
  for idx in range(box_index):
    other_box = boxes[idx]
    if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(center(other_box, "y") - center(box, "y")) < tolerance:
      return False
    if box["x"] > 1000or box["y"] > 1000
      return False
 
  return True
 
# Extract files from directory
def get_files(pattern):
  files = []
 
  # For each file that matches the specified pattern
  for file_name in glob.iglob(pattern, recursive=True):
 
    # Add the image file to the list of files
    files.append(file_name)
 
  # Return the complete file list
  return files
     
# Download pre-trained model and save to hard-drive
def load_model(model_name):
  url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + model_name + '.tar.gz'
     
  # Download a file from a URL that is not already in the cache
  model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)
 
  print("Model path: ", str(model_dir))
   
  model_dir = str(model_dir) + "/saved_model"
  model = tf.saved_model.load(str(model_dir))
 
  return model

# Load images in RGB
def load_rgb_images(pattern, shape=None):
  # Get a list of all the image files in a directory
  files = get_files(pattern)
 
  # For each image in the directory, convert it from BGR format to RGB format
  images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
 
  # Resize the image if the desired shape is provided
  if shape:
    return [cv2.resize(img, shape) for img in images]
  else:
    return images
 
# Load SSD architecture model with COCO dataset
def load_ssd_coco():
  return load_model("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")

# Label and save image cropped
def save_image_annotated(img_rgb, file_name, output, model_traffic_lights=None):
  # Create annotated image file 
  output_file = file_name.replace('.jpg', '_test.jpg')
     
  # For each bounding box that was detected  
  for idx in range(len(output['boxes'])):
 
    # Extract the type of the object that was detected
    obj_class = output["detection_classes"][idx]
     
    # How confident the object detection model is on the object's type
    score = int(output["detection_scores"][idx] * 100)
         
    # Extract the bounding box
    box = output["boxes"][idx]
 
    color = None
    label_text = ""
 
    if obj_class == LABEL_PERSON:
      color = (255, 100, 0)
      label_text = "Person " + str(score)
    if obj_class == LABEL_CAR:
      if score >= 50:
        color = (255, 255, 0)
        label_text = "Car " + str(score)
    if obj_class == LABEL_BUS:
      color = (255, 255, 0)
      label_text = "Bus " + str(score)
    if obj_class == LABEL_TRUCK:
      color = (255, 255, 0)
      label_text = "Truck " + str(score)
    if obj_class == LABEL_STOP_SIGN:
      color = (255, 20, 20)
      label_text = "Stop Sign " + str(score)
    if obj_class == LABEL_TRAFFIC_LIGHT:
      color = (255, 255, 255)
      label_text = "Traffic Light " + str(score)
             
      if model_traffic_lights:
       
              # Annotate the image and save it
        img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
        img_inception = cv2.resize(img_traffic_light, (299, 299))
         
                # Uncomment this if you want to save a cropped image of the traffic light
        #cv2.imwrite(output_file.replace('.jpg', '_crop.jpg'), cv2.cvtColor(img_inception, cv2.COLOR_RGB2BGR))
        img_inception = np.array([preprocess_input(img_inception)])
 
        prediction = model_traffic_lights.predict(img_inception)
        label = np.argmax(prediction)
        score_light = str(int(np.max(prediction) * 100))
        if label == 0:
          label_text = "Green " + score_light
        elif label == 1:
          label_text = "Yellow " + score_light
        elif label == 2:
          label_text = "Red " + score_light
        else:
          label_text = 'NO-LIGHT'  # This is not a traffic light
 
    if color and label_text and accept_box(output["boxes"], idx, 10.0) and score > 50:
      cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
      cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
  cv2.imwrite(output_file, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
  print(output_file)
 
def center(box, coord_type):
  """
  Get center of the bounding box.
  """
  return (box[coord_type] + box[coord_type + "2"]) / 2
 
def perform_object_detection(model, file_name, save_annotated=False, model_traffic_lights=None):
  """
  Perform object detection on an image using the predefined neural network.
  """
  # Store the image
  img_bgr = cv2.imread(file_name)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  input_tensor = tf.convert_to_tensor(img_rgb) # Input needs to be a tensor
  input_tensor = input_tensor[tf.newaxis, ...]
 
  # Run the model
  output = model(input_tensor)
 
  print("num_detections:", output['num_detections'], int(output['num_detections']))
 
  # Convert the tensors to a NumPy array
  num_detections = int(output.pop('num_detections'))
  output = {key: value[0, :num_detections].numpy()
            for key, value in output.items()}
  output['num_detections'] = num_detections
 
  print('Detection classes:', output['detection_classes'])
  print('Detection Boxes:', output['detection_boxes'])
 
  # The detected classes need to be integers.
  output['detection_classes'] = output['detection_classes'].astype(np.int64)
  output['boxes'] = [
    {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
     "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]
 
  if save_annotated:
    save_image_annotated(img_rgb, file_name, output, model_traffic_lights)
 
  return img_rgb, output, file_name
     
def perform_object_detection_video(model, video_frame, model_traffic_lights=None):
  """
  Perform object detection on a video using the predefined neural network.
     
  Returns the annotated video frame.
  """
  # Store the image
  img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
  input_tensor = tf.convert_to_tensor(img_rgb) # Input needs to be a tensor
  input_tensor = input_tensor[tf.newaxis, ...]
 
  # Run the model
  output = model(input_tensor)
 
  # Convert the tensors to a NumPy array
  num_detections = int(output.pop('num_detections'))
  output = {key: value[0, :num_detections].numpy()
            for key, value in output.items()}
  output['num_detections'] = num_detections
 
  # The detected classes need to be integers.
  output['detection_classes'] = output['detection_classes'].astype(np.int64)
  output['boxes'] = [
    {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
     "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]
 
  # For each bounding box that was detected  
  for idx in range(len(output['boxes'])):
 
    # Extract the type of the object that was detected
    obj_class = output["detection_classes"][idx]
     
    # How confident the object detection model is on the object's type
    score = int(output["detection_scores"][idx] * 100)
         
    # Extract the bounding box
    box = output["boxes"][idx]
 
    color = None
    label_text = ""
 
    if obj_class == LABEL_PERSON:
      color = (0, 255, 255)
      label_text = "Person " + str(score)
    if obj_class == LABEL_CAR:
      color = (255, 255, 0)
      label_text = "Car " + str(score)
    if obj_class == LABEL_BUS:
      color = (255, 255, 0)
      label_text = "Bus " + str(score)
    if obj_class == LABEL_TRUCK:
      color = (255, 255, 0)
      label_text = "Truck " + str(score)
    if obj_class == LABEL_STOP_SIGN:
      color = (128, 0, 0)
      label_text = "Stop Sign " + str(score)
    if obj_class == LABEL_TRAFFIC_LIGHT:
      color = (255, 255, 255)
      label_text = "Traffic Light " + str(score)
             
      if model_traffic_lights:
       
              # Annotate the image and save it
        img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
        img_inception = cv2.resize(img_traffic_light, (299, 299))
         
        img_inception = np.array([preprocess_input(img_inception)])
 
        prediction = model_traffic_lights.predict(img_inception)
        label = np.argmax(prediction)
        score_light = str(int(np.max(prediction) * 100))
        if label == 0:
          label_text = "Green " + score_light
        elif label == 1:
          label_text = "Yellow " + score_light
        elif label == 2:
          label_text = "Red " + score_light
        else:
          label_text = 'NO-LIGHT'  # This is not a traffic light
 
    # Use the score variable to indicate how confident we are it is a traffic light (in % terms)
    # On the actual video frame, we display the confidence that the light is either red, green,
    # yellow, or not a valid traffic light.
    if color and label_text and accept_box(output["boxes"], idx, 20.0) and score > 20:
      cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
      cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
  output_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  return output_frame
 
def double_shuffle(images, labels):
  """
  Shuffle the images to add some randomness.
  """
  indexes = np.random.permutation(len(images))
 
  return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]
 
def reverse_preprocess_inception(img_preprocessed):
  """
  Reverse the preprocessing process.
  """
  img = img_preprocessed + 1.0
  img = img * 127.5
  return img.astype(np.uint8)