# AI-Assignment-Footfall-Counter-Jayanth-Chatrathi
Footfall Counter using YOLO + DeepSORT
1. Brief Description of the Approach
This project implements a computer vision-based footfall counting system that detects, tracks, and counts the number of people entering and exiting through a defined Region of Interest (ROI). The system uses the YOLOv8 object detection model to identify humans in video frames and DeepSORT for multi-object tracking to maintain consistent identities across frames, even during occlusions. Each detected person is assigned a unique ID, and their trajectory is analyzed to determine whether they crossed the ROI line. The approach also includes visual overlays of bounding boxes, trajectory trails, and an optional heatmap of movement density. Additionally, a FastAPI endpoint is provided for automated processing.
2. Video Source Used
The demonstration video was sourced from a publicly available crowd surveillance video on YouTube. Alternatively, a local video file such as 'yolo_test.mp4' can be used for testing the system.
3. Explanation of Counting Logic
1. A virtual ROI line is defined across the video frame.
2. For each tracked person, their centroid position is computed in each frame.
3. The algorithm calculates the relative position of the centroid with respect to the ROI line using a cross-product sign.
4. When a track’s centroid changes sides (sign flips) and the line segments intersect the ROI, it is registered as a crossing event.
5. The direction of motion is determined by projecting the velocity vector onto the ROI’s normal direction.
6. Depending on the direction, the counter increments either 'IN' or 'OUT'.
7. Each track ID is assigned a short cooldown period to avoid double-counting.
4. Dependencies and Setup Instructions
### Dependencies (install via requirements.txt):
ultralytics==8.2.67
opencv-python==4.10.0.84
numpy==1.26.4
deep-sort-realtime==1.3.2
tqdm==4.66.4
yt-dlp==2024.11.4
fastapi==0.115.4
uvicorn==0.32.0
# Setup Instructions:
1. Clone or copy the repository files.
2. Create and activate a Python virtual environment.
3. Install dependencies using:
   pip install -r requirements.txt
4. Run the script on a video file:
   python footfall_counter.py --source path_to_video.mp4 --roi 100,400,1100,400 --heatmap --show
5. To start the FastAPI server:
   python footfall_counter.py --api
6. Use Postman or curl to upload and process videos.
5. Output
The processed video includes bounding boxes, track IDs, entry/exit counters, and an optional heatmap overlay. Final counts are displayed both on the video and in JSON format after processing.
