import cv2
import os

def capture_image():
	# Open the webcam
	cap = cv2.VideoCapture(0) #0 indicates the defult camera

	if not cap.isOpened():
		print('Error: Could not open webcam')
		return None

	# Capture frame-by-frame
	ret, frame = cap.read()

	# Release the webcam
	cap.release()

	if not ret:
		print('Error: Failed to capture image')
		return None

	# Ensure directory exists
	save_dir = 'webcam'
	os.makedirs(save_dir, exist_ok=True)

	# Save the captured image
	save_path = os.path.join(save_dir, 'img.png')
	cv2.imwrite(save_path, frame)

	return frame

def capture_vid():
	cap = cv2.VideoCapture(0) # Capture defult webcam (0)

	# Error if webcam not opened
	if not cap.isOpened():
		print('Error: Could not access webcam')
		return

	# Define the code and rceate a VideoWriter object
	save_dir = 'webcam'
	os.makedirs(save_dir, exist_ok=True)
	save_path = os.path.join(save_dir, 'output.mp4')

	# Define the code and specificy output format
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') #Code for MP4
	fps = 20.0
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

	print("Press 'q' to stop recording")

	while True:
		ret, frame = cap.read()
		if not ret:
			print('Failed to capture frame. Exiting...')
			break

		# Write the frame to the video file
		out.write(frame)

		cv2.imshow('Live Feed', frame)

		# Break the loop when q is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release webcam and video writer
	cap.release()
	out.release()
	cv2.destroyAllWindows()

	print(f"Video saved at {save_path}")





# Example usage
# image = capture_image()

vid = capture_vid()

# if image is not None:
# 	cv2.imshow('Captured Image', image)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

