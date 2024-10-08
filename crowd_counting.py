""" Starting from the U-Net result on images (i.e. binary images with circles
that indicates head of people), remove noise, count the number of people in 
the image using of between:
 - distance transform. 
 - countour detection
Then save the count in a csv file. """

import cv2 as cv
import os
from scipy.ndimage import maximum_filter
import numpy as np
from tqdm import tqdm

IMAGE_DIR = 'output/unet-train700'
LABEL_DIR = 'input/labels_backup'
# IMAGE_DIR = 'output/unet-val700'
# LABEL_DIR = 'data/belmonte_venice_img/Venezia_cc/train_data/ground_truth'

NEIGHBORHOOD_SIZE = 3
# MIN_DISTANCE = 0.2

def main():
    # csv_file = open('output/crowd_count_distance.csv', 'w')
    # csv_file.write("Min distance,Mean square error\n")

    # for MIN_DISTANCE in tqdm(np.arange(0.9, 1, 0.1)):
    #     mean_square_error = count_people(MIN_DISTANCE)
    #     print(f"Mean square error with MIN_DISTANCE={MIN_DISTANCE:.2f}: {mean_square_error}")
    #     csv_file.write(f"{MIN_DISTANCE},{mean_square_error}\n")
    
    # csv_file.close()

    # ##################

    # csv_file = open('output/crowd_count_contour.csv', 'w')
    # csv_file.write("Min area,Mean square error\n")

    # for min_area in tqdm(np.arange(0, 0.1, 0.01)):
    #     mean_square_error = count_people_contour(min_area)
    #     print(f"Mean square error with MIN_DISTANCE={min_area}: {mean_square_error}")
    #     csv_file.write(f"{min_area},{mean_square_error}\n")
    
    # csv_file.close()
    
    
    count_people_contour(0)
    

def count_people_contour(min_area):
    os.makedirs('output/crowd_count', exist_ok=True)
    csv_file = open(f'output/crowd_count/min_contour_area={min_area:.2f}.csv', 'w')
    csv_file.write("Image,Count,True\n")

    square_error = 0

    for img_path in os.listdir(IMAGE_DIR):
        img = cv.imread(os.path.join(IMAGE_DIR, img_path), cv.IMREAD_GRAYSCALE)
        
        # Remove noise by eroding the image (= only bigger circles remain)
        img = cv.erode(img, None, iterations=1)

        # Find contours
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        big_contours = [c for c in contours if cv.contourArea(c) > min_area]

        # Get the true count of people in the image and update the square error
        label_csv = open(os.path.join(LABEL_DIR, img_path.replace('.jpg', '.csv')))
        true_count = len(label_csv.readlines())-1

        square_error += (true_count - len(big_contours))**2

        # Save the count in a csv file
        csv_file.write(f"{img_path},{len(big_contours)},{true_count}\n")
    
    csv_file.close()
    mean_square_error = square_error / len(os.listdir(IMAGE_DIR))
    return mean_square_error


def count_people_dist_func(MIN_DISTANCE):
    os.makedirs('output/crowd_count', exist_ok=True)
    csv_file = open(f'output/crowd_count/min_distance={MIN_DISTANCE:.2f}.csv', 'w')
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

        # Get the true count of people in the image and update the square error
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