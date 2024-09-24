""" Starting from the U-Net result on images (i.e. binary images with circles
that indicates head of people), remove noise, count the number of people in 
the image using distance transform. Then save the count in a csv file. """

import cv2 as cv
import os
from scipy.ndimage import maximum_filter
import numpy as np
from tqdm import tqdm

IMAGE_DIR = 'output/unet-train700'
LABEL_DIR = 'input/labels_backup'
NEIGHBORHOOD_SIZE = 3
# MIN_DISTANCE = 0.2

def main():
    # csv_file = open('output/crowd_count.csv', 'w')
    # csv_file.write("Min distance,Mean square error\n")

    # for MIN_DISTANCE in tqdm(np.arange(0.9, 1, 0.1)):
    #     mean_square_error = count_people(MIN_DISTANCE)
    #     print(f"Mean square error with MIN_DISTANCE={MIN_DISTANCE:.2f}: {mean_square_error}")
    #     csv_file.write(f"{MIN_DISTANCE},{mean_square_error}\n")
    
    # csv_file.close()
    count_people(0.2)
    

def count_people(MIN_DISTANCE):
    os.makedirs('output/crowd_count', exist_ok=True)
    csv_file = open(f'output/crowd_count/{MIN_DISTANCE:.2f}.csv', 'w')
    csv_file.write("Image,Count,True\n")

    square_error = 0

    for img_path in os.listdir(IMAGE_DIR):
        img = cv.imread(os.path.join(IMAGE_DIR, img_path), cv.IMREAD_GRAYSCALE)
        
        # Remove noise by eroding the image (= only bigger circles remain)
        img = cv.erode(img, None, iterations=1)

        # Calculate distance transform
        dist_transform = cv.distanceTransform(img, cv.DIST_L2, 3)
        cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)

        # Find local maxima
        local_max = maximum_filter(dist_transform, size=NEIGHBORHOOD_SIZE) == dist_transform

        # Create a mask of local maxima
        mask = np.zeros(dist_transform.shape, dtype=bool)
        mask[local_max] = True

        coords = np.argwhere(mask)
        coords = [c for c in coords if dist_transform[c[0], c[1]] > MIN_DISTANCE]

        # Get the true count of people in the image
        label_csv = open(os.path.join(LABEL_DIR, img_path.replace('.jpg', '.csv')))
        true_count = len(label_csv.readlines())-1

        square_error += (true_count - len(coords))**2

        # print(f"{img_path}: {len(coords)} (True: {true_count})")

        # Save the count in a csv file
        csv_file.write(f"{img_path},{len(coords)},{true_count}\n")

    csv_file.close()
    mean_square_error = square_error / len(os.listdir(IMAGE_DIR))
    return mean_square_error

if __name__ == "__main__":
    main()