Instructions
This project contains a Python file (21098082_roadsegmentation.py) that performs road segmentation on a set of images using a color-based segmentation algorithm. The results of the segmentation are evaluated against ground truth masks provided for each image. The evaluation metrics include accuracy, precision, recall, and F1 score.

Folder Structure
Ensure that the following files and folders are in the same directory:
•	21098082_roadsegmentation.py: The Python script containing the segmentation algorithm and evaluation code.
•	images/: A folder containing the test images (e.g., 0.png, 1.png, 2.png, ..., 30.png).
•	masks/: A folder containing the ground truth masks (e.g., 0.png, 1.png, 2.png, ..., 30.png).
•	roads_segmentation.csv: An Excel file containing the list of image indices and their corresponding file names.

Usage
Run the Python file (21098082_roadsegmentation.py) to perform the segmentation and evaluation.
The script will:
1.	Read the test images from the images/ folder.
2.	Read the ground truth masks from the masks/ folder.
3.	Perform color-based segmentation on each image.
4.	Evaluate the segmentation results against the ground truth masks.
5.	Print the evaluation metrics (accuracy, precision, recall, and F1 score) for each image and the average metrics across all images to the console.

Image Representation
To observe the image representation result for a specific image, you can specify the index of the image you want within the script. By default, the script is set to display the results for the first image (index 0). Modify the script as needed to display results for other images.
 
This will display the original image, image in HSV color model, the region mask, and the ground truth mask for the specified image index.

Output
The evaluation results for each image and the average evaluation metrics will be printed to the console. The script will provide detailed metrics for accuracy, precision, recall, and F1 score for each image, as well as their average values across all images.


