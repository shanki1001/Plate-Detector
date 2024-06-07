import cv2
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLO model
try:
    model = YOLO("yolov8m.pt")
    names = model.model.names
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit(1)

# Open the video file
cap = cv2.VideoCapture("cars8.mp4")
if not cap.isOpened():
    logging.error("Error reading video file")
    exit(1)

# Get video properties
try:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
except Exception as e:
    logging.error(f"Error getting video properties: {e}")
    cap.release()
    exit(1)

# Initialize the video writer
try:
    video_writer = cv2.VideoWriter(
        "speed_estimation.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (w, h)
    )
except Exception as e:
    logging.error(f"Error initializing video writer: {e}")
    cap.release()
    exit(1)

# Define the line points for speed estimation
line_pts = [(0, h // 2), (w, h // 2)]

# Initialize the speed estimation object
try:
    speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=True)
except Exception as e:
    logging.error(f"Error initializing SpeedEstimator: {e}")
    cap.release()
    video_writer.release()
    exit(1)

# Process the video frames
try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Video frame is empty or video processing has been successfully completed.")
            break
        
        # Track objects in the frame
        try:
            tracks = model.track(im0, persist=True, show=False)
        except Exception as e:
            logging.error(f"Error during object tracking: {e}")
            continue

        # Estimate the speed of the tracked objects
        try:
            im0 = speed_obj.estimate_speed(im0, tracks)
        except Exception as e:
            logging.error(f"Error during speed estimation: {e}")
            continue
        
        # Ensure the frame has the correct dimensions before writing to the output video
        if im0.shape[1] != w or im0.shape[0] != h:
            im0 = cv2.resize(im0, (w, h))
        
        # Write the processed frame to the output video
        try:
            video_writer.write(im0)
        except Exception as e:
            logging.error(f"Error writing frame to video: {e}")
            break
finally:
    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()