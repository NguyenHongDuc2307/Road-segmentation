# Satelite Images Road Segmentation

## Problem
Implement and use computer vision methods to segment the roads in the satellite image [road.png] using Python, and numpy.

## Solution
A per-pixel-classification technique is used to build the output binary mask, that is, each pixel is independently classified as a part of a road or not.

To classify pixel (x, y), the pixels contained in the surrounding window of a predefined side length 'L' centered at (x, y) will be used as features, so that the input features vector will contain the (r, g, b) values of all these pixels (including the current target pixel at (x, y)), which makes the features vector size = 3xLxL. Here, we have L = 5.

As a classifier, a deep neural network of multiple hidden layers is used. The output layer consists of two neurons representing the two output classes in one-hot vector representation.

## Implementation
The described solution is implemented in python using Tensorflow (version 2.3.1), pillow (version 4.1.0) and sklearn (version 0.22.2.post1). It consists of three separate python scripts with the following purposes:
- Converting the image data set into "data.csv" text file used for training and testing.
- Building a classifier model, training it using the generated csv text file, and saving it in "model" directory.
- Classifying an input image using the already training classifier.


- The first script "convertToFeatureFiles.py" converts the image data set into csv data files that can be fed into the classifier for training and testing. 
To obtain good results, the ratio between the samples of the two classes shouldn't be too large, and naturally the ratio between non-road pixels and road pixels in an image is very large, that's why dropout must be performed to balance the dataset, but it must be performed in a random fashion to ensure that the classifier is exposed to a diverse set of samples from each class. To perform the dropout for each image, the script loads the image along with its corresponding expected output, then it divides it into two sets of pixels, road pixels and non-road pixels, randomly shuffles both sets, and then for each road pixel, we take two non-road pixels (which ensures the ratio between road samples and non-road samples to be 1:2 in the output csv file), generate their feature vectors as mention earlier, and write them to the output file. This is done for input and output images to generate "data.csv".

- The second script "train.py" is to train the classifier. The second script uses Tensorflow to initialize a deep neural network of three hidden layers of sizes 100, 50, 30 neurons respectively, then loads data.csv files, and use it to train the neural network for a predefined number of epochs, the testing data is used to evaluate the accuracy of the model at the end of the training. After the training finishes, the script saves the model on the file system in "model" directory that can be loaded to recalculate its accuracy, or use it in classifying input images.

- The third script "classify.py" is for the classification of an input image, it takes the names of the input and output image names as command line arguments, it loads the model and the input image from the 'image-input' directory, generate the feature vector for each pixel, classify it using the loaded classifier, and generate the output image with zero/one value for each pixel based on the classifier prediction for that pixel, then it saves the generated image in 'image-output' directory using the output image name entered in the command line arguments.

## CSV file format
In the current implmentation the window size is 5 pixels.

The csv files follow a very simple format:
- each record (line) is a single feature vector with its ground truth outcome.
- each feature vector consists of the RGB values of pixels contained in a 5x5 window (three integers for each pixel), and a binary value representing the class of the central pixel of that window.

Accordingly, each line in the csv files should consist of (3x5x5) + 1 integers as follows:
```
r1, g1, b1, r2, g2, b2, r3, g3, b3, . . . . . , r25, g25, b25, y
```

Where r13, g13, b13, are the RGB values of the central pixel, and y is the class of the central pixel.

The pixels in the window are ordered in the feature vector in row-major order.

## Dataset
[www.cs.toronto.edu/~vmnih/data/](https://www.cs.toronto.edu/~vmnih/data/)

### Dataset size
As mentioned earlier, a per-pixel-classification technique is used, which means that a single pixel generates one feature vector, so an 1500x1500 image will generate 2250000 vectors, which is too large to be handled by a personal computer, that's why the 'convertToFeatureFiles.py' script generates only 200000 vectors per file.

## Usage
- Prepare the dataset in the directories contained in 'airs-dataset' directory. Each input image must have a corresponding binary (black and white) output image of the same exact dimensions in the corresponding output directory.
- Run 'convertToFeatureFiles.py'. It should generate "data.csv" file. 
- Run 'train.py'. It should build the NN, and train it using the generated files in the previous step.
- Prepare the input image in 'image-input' directory.
- Run 'classify.py' giving it two cmd arguments, the name if the input image file from the previous step, and the desired name of the output image file. It should generate the corresponding binary mask in 'image-output' directory.

Note: The model is already available in "model" directory. You can only run "classify.py" to validate imput images without building a new model.

## Samples
#### Sample input
![sample input](/image-input/road3.png)

#### Expected output
![expected output](/image-output/road3_true.png)
#### Output
![output](/image-output/road3.png)

## Tools
- Python version 3.5.4
- [Tensorflow](https://www.tensorflow.org/) version 2.3.1
- [pillow](https://python-pillow.org/) version 4.1.0
- [sklearn] (https://scikit-learn.org/stable/) version 0.22.2.post1



### Discussion
The features considered in this model are the RGB values of the pixels surrounding each pixel, which are not enough because road pixels and non-road pixels made of the same material looks the same, for example road pixels and buildings or parking lots. The validation accuracy is about 85%.

In fact, the model doesn't work well with images in "road images" directory provided by company because they are not in the same dataset. We can verify this with the sample [sample input](/image-input/road1.png). 

A possible solution is to add a convolutional neural network as another layer to the model before the used neural network, the added CNN can segment the building based on other features and exclude them before further processing. Since  the deadline has come, I did not have time to learn CNN and try to implement this solution yet.
