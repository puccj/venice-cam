from UNet.unet import UNet

import os
import matplotlib.pyplot as plt
import cv2
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/700.pth'

image_dir = "/home/daniele/U-Net/data/val_images"  # Folder with images to predict on
output_dir = "output/unet-val700"  # Folder to save predicted masks


# Define the same validation transform used during training
def get_transform(image_shape=None):
    if image_shape is None:
        image_shape = (IMAGE_HEIGHT*8, IMAGE_WIDTH*8)
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

    image_np = image.permute(1, 2, 0).cpu().numpy()
    # image_np = (image_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_dir}/original-{image_path.split("/")[-1]}', image_np)

    return image.unsqueeze(0)  # Add batch dimension

def predict_image(model, image_tensor, device=DEVICE):
    image_tensor = image_tensor.to(device).float()
    with torch.no_grad():  # No need to track gradients during inference
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)  # Apply sigmoid for binary segmentation
        prediction = (prediction > 0.5).float()  # Threshold the prediction
    return prediction.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array

def main():
    model = UNet()
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(DEVICE)

    os.makedirs(output_dir, exist_ok=True)

    loop = tqdm(os.listdir(image_dir))

    for image_file in loop:
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Check for valid image formats
            image_path = os.path.join(image_dir, image_file)
            
            # print(image.shape)
            # Preprocess the image
            image_tensor = preprocess_image(image_path).float()
            
            # Get prediction from the model
            predicted_mask = predict_image(model, image_tensor)

            # Optionally visualize or save the prediction
            plt.imsave(f"{output_dir}/{image_file}", predicted_mask, cmap='gray')

            # update the progress bar
            loop.set_description(f"Predicting {image_file}")

if __name__ == '__main__':
    main()