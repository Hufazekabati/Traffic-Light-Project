
import cv2 # Computer vision library
import object_detection # Contains methods for object detection in images
 
# Get a list of jpeg image files containing traffic lights
files = object_detection.get_files('traffic_light_input/*.jpg')
 
# Load the object detection model
this_model = object_detection.load_ssd_coco()
 
# Keep track of the number of traffic lights found
traffic_light_count = 0
 
# Keep track of the number of image files that were processed
file_count = 0
 
# Display a count of the number of images we need to process
print("Number of Images:", len(files))
 
# Go through each image file, one at a time
for file in files:
 
  # Detect objects in the image
  # img_rgb is the original image in RGB format
  # out is a dictionary containing the results of object detection
  # file_name is the name of the file
  (img_rgb, out, file_name) = object_detection.perform_object_detection(model=this_model, file_name=file, save_annotated=None, model_traffic_lights=None)
     
  # Every 10 files that are processed
  if (file_count % 10) == 0:
 
    # Display a count of the number of files that have been processed
    print("Images processed:", file_count)
 
    # Display the total number of traffic lights that have been identified so far
    print("Number of Traffic lights identified: ", traffic_light_count)
         
  # Increment the number of files by 1
  file_count = file_count + 1
 
  # For each traffic light (i.e. bounding box) that was detected
  for idx in range(len(out['boxes'])):
 
    # Extract the type of object that was detected  
    obj_class = out["detection_classes"][idx]
         
    # If the object that was detected is a traffic light
    if obj_class == object_detection.LABEL_TRAFFIC_LIGHT:
         
      # Extract the coordinates of the bounding box
      box = out["boxes"][idx]
             
      # Extract (i.e. crop) the traffic light from the image     
      traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
             
      # Convert the traffic light from RGB format into BGR format
      traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)
 
      # Store the cropped image in a folder named 'traffic_light_cropped'     
      cv2.imwrite("traffic_light_cropped/" + str(traffic_light_count) + ".jpg", traffic_light)
             
      # Increment the number of traffic lights by 1
      traffic_light_count = traffic_light_count + 1
 
# Display the total number of traffic lights identified
print("Number of Traffic lights identified:", traffic_light_count)