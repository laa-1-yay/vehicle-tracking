# Vehicle-detection-and-tracking

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog.png
[image4]: ./examples/sliding_window.png
[image6]: ./examples/bboxes_and_heat.png
[image7]: ./examples/HOC.png
[image8]: ./examples/spatial_binning.png
[video1]: ./project_video.mp4

The whole code is present in the file `vehicle_tracking.ipynb`

### Color Histogram:

The color channels of an image in a color space can be divided into bins, which represent color combinations of a car or notcar image to train classifier on. 

Here is an example of RGB channel color histogram :   


![alt text][image7]

   
### Spatial binning:

As cars can be of many different colors and hues, the color histogram is not very useful. The spatial appearance of vehicle in an image is a useful metric that can be used to build feature for training. The car/notcar image is resized and the value of pixels are stored as feature sets. 

Here is an example:    

![alt text][image8]
     

### Histogram of Oriented Gradients (HOG)

#### 1. HOG feature extraction from the training images.

The code for this step is contained in the 9th code cell of the IPython notebook under the heading `extract_features`  

I started by reading in all the `vehicle` and `non-vehicle` images and then shuffling them.  Here are examples of each of the `vehicle` and `non-vehicle` classes:    

![alt text][image1]   

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:   

![alt text][image2]    
     
#### 2. Explain how you settled on your final choice of HOG parameters.

Various parameters can be adjusted to get better performance such as :

| Parameter     | Details | Examples |
|:-------------:|:-------------:|:-------------:|
| spatial_size | Spatial binning dimensions| (32,32) or (16,16) |
| hist_bins    | Number of histogram bins | 16, 32, 40, ...|
| color_space  | Color Space Conversion | 'RGB', 'YUV', 'YCrCb', 'HSV', 'HLS', 'LUV' |
| orient       | HOG orientation resolution | 9, 11, 13, ... |
| pix_per_cell | HOG Pixels per cell | 8, 12, ...|
| cell_per_block | HOG cells per block | 2, 3, ...|
| hog_channel  | channel to be applied in HOG | 0, 1, 2, or 'ALL' |


After experimenting with a lot of parameters and measuring the accuracy and evaluating the final annotated video I settled on the following paramters:

| Parameter    | Value |
|:------------:|:-------------:|
| spatial_size | (32,32) |
| color_space  | 'YCrCb' |
| orient       | 9 |
| pix_per_cell | 8 |
| cell_per_block | 2 |
| hog_channel  | 'ALL' |

It makes more sense to use (8x8) pixels per cell for a 64x64 training image. After trying different orientations like 6, 8 and 9, all of giving reasonable results, the alignment of gradients in hog visualization with orientation, 9 was able to clearly differentiate between car and non -cars. So, I chose 9 as orientation.


#### 3. Training of  classifier using your selected HOG features (and color features if you used them).


The code for training classifier is provided in code cell (18 -21) under the heading `Training the classifier`. In code cell 18, features are extracted from training data using extract_features which uses hog features , binned color features and histogram of all color channelst to compute a 1-D feature array for a training image. 

The concatenated feature data are normalized by calling `StandardScaler` from `sklearn.preprocessing` library and shuffled and divided into training and test dataset in `train_test_split()` function.     

In code cell 19, data is normalized using StandardScaler() and then data is divided into training (80%) and testing data (20%).   

In code cell 21, LinearSVC was used to train the data . It took about 19.95 seconds to train SVC for feature vector length of 8460. After that, trained classifier is tried on test data to compute accuracy. I played with different parameters like color_space , hog_channel . My observations are given below :-    

With color_space = 'RGB' and hog_channel = 0 gave Test_Accuracy = 97.4 %, Feature vector length = 4932     
With color_space = 'YCrCb' and hog channel = ALL gave Test_Accuracy = 97.62 % , Feature vector length = 4932     
With color_space = 'RGB' and hog_channel = 0 gave Test_Accuracy = 98.14 %, Feature vector length = 8460    
With color_space = 'YCrCb' and hog channel = ALL gave Test_Accuracy = 99.01 %, Feature vector length = 8460    
    
I finally settled on color_space = YCrCb and hog channel =ALL and it gave me accuracy = 99.01%     

The classifer training log is as follows.

```python
No. of data points 17760
No. of features 8460
Feature vector length: 8460
10.99 Seconds to train SVC...
Test Accuracy of SVC =  0.9901
```

### Sliding Window Search

#### 1. Implemention of sliding window search.  Scales to search and how much to overlap windows?

I used find_cars function in the 23rd cell, which first extracted all the hog features from an image and then subsampled the image to divide it into different overlaying windows for which the trained classifer LinearSVC was used to predict whether the car existed in a window and if vehicle is detected.

An image is then given as output with a boxes drawn over the vehicle. Scale parameter was used to select the size of sliding window and cell per step defined the overlap between windows. 

Cell per step of 2 was used which is equivalent to window overalap of 75%. This is faster than method which first finds windows to search using siding window and then computes hog features from each search window.

#### 2. Examples of test images demonstrating working of pipeline.  Optimization the performance of your classifier.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images of the windows drawn:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Filter for false positives and method for combining overlapping bounding boxes.

The process_image() function in cell number 25 explains the pipeline which employs functions apply_threshold(), add_heat() and draw_labeled_bboxes() which are defined in cell 22 which apply thresholding to heat maps removing most of the false positives, and draws bounding boxes around these using the label feature from scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. Each blob corresponded to a vehicle.

Here's an example result showing the heatmap from a series of frames of video, and the final output with the bounding boxes then overlaid on the frames of video .  

`heatmap_threshold` is set to one. Therefore, you can observe in the last image the false positive on the extreme left of the image (other lane) is avoided since only one window is used to enclose it. Therefore, threshold is applied and it is not shown in the final ouput.

### Here are six frames and their corresponding heatmaps:
    
![alt text][image6]    


---

### Discussion

#### 1. Problems / issues in implementation of this project and improvements -

Problems :

- False positives for objects like trees.
- Failure for non trained objects like pedestrians. 
- Detection of white car is difficult in the bright color patch of the road.
- Failure under shadows, bright roads, night, rain, snow, fog.

Improvements:

- Experimenting with different classifier. E.g. Decision Tree
- Do better in presence of shadows, bright light, and night.
- Use previous video frames and keep track of detected vehicles using averaging technique on previous frames.
- Train on multiple types of vehicles like cycles, trucks, motorbikes to simplify learning and classification.
- Augmentation of dataset for better results.
- Stabilizing the disturbance in the detected boxes.
