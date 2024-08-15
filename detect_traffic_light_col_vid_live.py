import cv2 
import numpy as np 
import object_detection  
from tensorflow import keras  
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os

# Load the SSD neural network that is trained on the COCO data set
model_ssd = object_detection.load_ssd_coco()

# Load the trained neural network for traffic lights (or any other specific object)
model_traffic_lights_nn = keras.models.load_model("traffic.keras")

def main():
    # Open the default webcam (0 indicates the first camera)
    cap = cv2.VideoCapture(0)
    
    # Error check if webcam is opened
    if not cap.isOpened():
        print('Error: Could not access webcam')
        return
    
    # Optionally save the annotated video
    save_dir = 'webcam'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'output_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 codec
    fps = 20.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    print("Press 'q' to stop recording")
    
    while True:
        # Capture frame-by-frame from the webcam
        success, frame = cap.read()
        
        if not success:
            print('Failed to capture frame. Exiting...')
            break
        
        # Process and annotate the frame
        output_frame = object_detection.perform_object_detection_video(
            model_ssd, frame, model_traffic_lights=model_traffic_lights_nn)
        
        # Display the annotated frame in real-time
        cv2.imshow('Live Feed Annotated', output_frame)
        
        # Optionally save the annotated frame
        out.write(output_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Annotated video saved at {save_path}")

# Run the main function
main()
