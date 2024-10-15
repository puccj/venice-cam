import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from ultralytics import YOLO

from UNet.unet import UNet
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import maximum_filter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
# model = YOLO("yolov8x.pt")

# Load UNet model
model = UNet()
checkpoint = torch.load('models/700.pth', weights_only=True)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(DEVICE)

# Open the video file
video_path = 'input/jetson2_video_20180628-161507.mp4'  # Path to your video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
times = []


# Define the same validation transform used during training
def get_transform(image_shape=None):
    smaller = min(image_shape[:2])
    # print(smaller)
    
    return A.Compose([
        # A.CenterCrop(smaller, smaller),
        # A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        ToTensorV2()
    ])

# Function to load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read the image
    image = cv2.resize(image, (image.shape[1] // 8, image.shape[0] // 8))
    transform = get_transform(image.shape)
    augmented = transform(image=image)
    image = augmented["image"]

    return image.unsqueeze(0)  # Add batch dimension


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # if frame_count == 1000:  # Process only first 100 frames
    #     break

    start_time = time.time()

    # Perform inference using YOLOv8
    # results = model.track(frame, persist=True, conf=0.5)

    # Perform inference using UNet
    frame = cv2.resize(frame, (frame.shape[1] // 8, frame.shape[0] // 8))
    transform = get_transform(frame.shape)
    augmented = transform(image=frame)
    image = augmented["image"].unsqueeze(0).to(DEVICE).float()
    with torch.no_grad(): 
        prediction = model(image)
        prediction = torch.sigmoid(prediction)  # Apply sigmoid for binary segmentation
        prediction = (prediction > 0.5).float()  # Threshold the prediction
    # convert the image to cv2 format
    img = prediction.squeeze().cpu().numpy() * 255
    img = img.astype(np.uint8)
    img = cv2.erode(img, None, iterations=1)
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    local_max = maximum_filter(dist_transform, size=3) == dist_transform
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[local_max] = True
    coords = np.argwhere(mask)
    coords = [c for c in coords if dist_transform[c[0], c[1]] > 0.2]


    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

    frame_count += 1

cap.release()

# Calculate the average processing time per frame
average_time = np.mean(times)
print(f"Average processing time per frame: {average_time:.5f} seconds")
print(f"Total frames processed: {frame_count}")

# Save time taken to process each frame
np.save('times.npy', times)

def plot_times(filename='times.npy'):
    import matplotlib.pyplot as plt

    times = np.load(filename)
    times = times[1:]   # Skip the first frame as it may have higher processing time

    plt.plot(times)
    plt.xlabel('Frame')
    plt.ylabel('Processing Time (ms)')
    plt.title('BG-subtraction algorithm computational time per frame (Rasperry)')
    plt.show()

    return np.mean(times)