import numpy as np
import cv2
from matplotlib import pyplot as pt
import pandas as pd

# Function to preprocess the input images
def preprocess_img(img):
    # Blur the images to remove noise
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.blur(img, (9, 9))
    return img

# Function for color segmentation
def color_segmentation(img, lower_hsv, upper_hsv, min_y):
    # Convert to HSV color model
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask to extract region between threshold
    region_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    # Limit the road to be below a certain y-coordinate
    region_mask[:min_y, :] = 0
    
    # Apply the mask to the original image
    segmented_img = cv2.bitwise_and(img_hsv, img_hsv, mask=region_mask)

    return segmented_img, region_mask

# Function to preprocess the groundtruth images
def preprocess_groundtruth(img):
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create mask by setting pixels with value 116 to 255, else 0
    ground_truth_mask = cv2.inRange(img, 116, 116)
    
    return ground_truth_mask

# Function to evaluate results
def evaluation(region_mask, ground_truth_mask):
    # Obtain true positive, true negative, false positive and false negative values
    true_positive = np.sum(np.logical_and(region_mask, ground_truth_mask))
    true_negative = np.sum(np.logical_and(np.logical_not(region_mask), np.logical_not(ground_truth_mask)))
    false_positive = np.sum(np.logical_and(region_mask, np.logical_not(ground_truth_mask)))
    false_negative = np.sum(np.logical_and(np.logical_not(region_mask), ground_truth_mask))
    
    # Calculate accuracy, precision, recall and f1 score
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score

# Load the CSV file
excel = pd.read_csv('roads_segmentation.csv')

# HSV thresholds
lower_hsv = np.array([0, 0, 80])   
upper_hsv = np.array([180, 60, 155]) 
min_y = 210  

# Initialize lists to store evaluation results
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Loop through the image index list in the CSV file
for index, row in excel.iterrows():
    image_path = row['image_name']
    mask_path = image_path.replace('images', 'masks')
    
    # Load the input image and ground truth image
    img = cv2.imread(image_path)
    ground_truth = cv2.imread(mask_path)
    
    if img is None or ground_truth is None:
        print(f"Error loading {image_path} or {mask_path}")
        continue
    
    # Preprocess the images
    img = preprocess_img(img)

    # Perform color segmentation with HSV thresholds
    segmented_img, region_mask = color_segmentation(img, lower_hsv, upper_hsv, min_y)
    
    # Preprocess the ground truth
    ground_truth_mask = preprocess_groundtruth(ground_truth)
    
    # Evaluate the results
    accuracy, precision, recall, f1_score = evaluation(region_mask, ground_truth_mask)
    
    # Append the results to the lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    
    # Print the evaluation results for each image
    print(f"Evaluation Results for {image_path}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print()
    
    # Plot the results for the image
    if index == 0:
        pt.figure()
        pt.subplot(2, 2, 1)
        pt.title("Original Image")
        pt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pt.axis('off') 
        pt.subplot(2, 2, 2)
        pt.title("HSV Image")
        pt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        pt.axis('off') 
        pt.subplot(2, 2, 3)
        pt.title("Region Mask")
        pt.imshow(region_mask, cmap='gray')
        pt.axis('off') 
        pt.subplot(2, 2, 4)
        pt.title("Ground Truth Mask")
        pt.imshow(ground_truth_mask, cmap='gray')
        pt.axis('off') 
        pt.show()
    
# Calculate the average of the evaluation results
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

# Print the average evaluation results
print("Average Evaluation Results:")
print(f"Accuracy: {avg_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1_score:.4f}")

