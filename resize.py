import cv2

# Input and output video paths
input_video_path = "6F9894AC-2BFC-48D3-BEE7-64B34751684F.mov"
output_video_path = "output_video_640x640.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 640))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))
    
    # Write the resized frame to the output video
    out.write(resized_frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video resizing to 640x640 complete. Saved as", output_video_path)
