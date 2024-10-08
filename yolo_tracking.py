import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import csv

import time

def save_trajectories_to_csv(trajectories, filename='trajectories.csv'):
    # Save the trajectories to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Frame", "X", "Y"])  # Column headers

        # Write each object's trajectory to the file
        for obj_id, trajectory in trajectories.items():
            for frame_index, point in enumerate(trajectory):
                x, y = point
                writer.writerow([obj_id, frame_index, x, y])

    print(f"Trajectories saved to {filename}")

def load_trajectories_from_csv(filename='trajectories.csv'):
    loaded_trajectories = {}

    # Load trajectories from CSV file
    with open('trajectories.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            obj_id = int(row[0])
            frame_index = int(row[1])
            x, y = int(row[2]), int(row[3])

            if obj_id not in loaded_trajectories:
                loaded_trajectories[obj_id] = []
            
            loaded_trajectories[obj_id].append((x, y))

    print("Trajectories loaded from 'trajectories.csv'")
    return loaded_trajectories


# Load the YOLO model (assuming you have YOLOv8 installed)
model = YOLO("yolov8x.pt")  # You can choose the yolov8n, yolov8s, etc., depending on your setup

# Open the video file
video_path = 'input/jetson2_video_20180628-161507.mp4'  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Store trajectories for each object
trajectories = {}

frame_count = 0
times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # if frame_count == 1000:  # Process only first 100 frames
    #     break

    start_time = time.time()

    # Perform tracking using YOLOv8
    # results = model.track(frame, persist=True, conf=0.5)
    results = model.predict(frame, conf=0.5)

    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

    # Iterate through the results
    # for result in results:
    #     for detection in result.boxes:  # Access boxes in each result
    #         id = int(detection.id)  # Unique ID for each object being tracked
    #         x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
    #         center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate center of the object

    #         # Store the trajectory of each object
    #         if id not in trajectories:
    #             trajectories[id] = []
    #         trajectories[id].append(center)

    frame_count += 1

cap.release()

# Calculate the average processing time per frame
average_time = np.mean(times)
print(f"Average processing time per frame: {average_time:.5f} seconds")
print(f"Total frames processed: {frame_count}")

# Save time taken to process each frame
np.save('times.npy', times)

# Save the trajectories to a CSV file
save_trajectories_to_csv(trajectories)

# Now, let's plot the tracked trajectories
plt.figure(figsize=(10, 8))
for obj_id, trajectory in trajectories.items():
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Object {obj_id}')

plt.title("Tracked Object Trajectories")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
# plt.legend()
plt.gca().invert_yaxis()  # Invert Y-axis to match video coordinate system
plt.show()
