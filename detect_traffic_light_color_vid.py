import cv2 
import numpy as np 
import object_detection  
from tensorflow import keras  
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
 
# Make sure the video file is in the same directory as your code
filename = 'NEWYORK.mp4'
file_size = (1920,1080) # Assumes 1920x1080 mp4
scale_ratio = 1 # Option to scale to fraction of original size. 
 
# We want to save the output to a video file
output_filename = 'las_vegas_annotated.mp4'
output_frames_per_second = 20.0
 
# Load the SSD neural network that is trained on the COCO data set
model_ssd = object_detection.load_ssd_coco()
 
# Load the trained neural network
model_traffic_lights_nn = keras.models.load_model("traffic.keras")
 
def main():
  
  # Load a video
  cap = cv2.VideoCapture(filename)
 
  # Create a VideoWriter object so we can save the video output
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  result = cv2.VideoWriter(output_filename,  
                           fourcc, 
                           output_frames_per_second, 
                           file_size) 
     
  # Process the video
  while cap.isOpened():
         
    # Capture one frame at a time
    success, frame = cap.read() 
         
    # Do we have a video frame? If true, proceed.
    if success:
         
      # Resize the frame
      width = int(frame.shape[1] * scale_ratio)
      height = int(frame.shape[0] * scale_ratio)
      frame = cv2.resize(frame, (width, height))
             
      # Store the original frame
      original_frame = frame.copy()
 
      output_frame = object_detection.perform_object_detection_video(
        model_ssd, frame, model_traffic_lights=model_traffic_lights_nn)
 
      # Write the frame to the output video file
      result.write(output_frame)
             
    # No more video frames left
    else:
      break
             
  # Stop when the video is finished
  cap.release()
     
  # Release the video recording
  result.release()
     
  # Close all windows
  cv2.destroyAllWindows() 
     
main()