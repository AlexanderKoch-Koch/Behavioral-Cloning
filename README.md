# Behavioral Cloning

## Model Architecture and Training Strategy

### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the network from NVIDIA. I thought this model might be appropriate because their neural net should do almost the same.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I added Dropout.
However, in the end I didn’t use a validation set because the loss of the network does not seem to show the performance in the simulator.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. These were the parts just before and after the bridge. To improve the driving behavior in these cases, I added more training data of these spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is a copy of the NVIDIA model.
My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 72-75).
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 71).
The model contains dropout layers in order to reduce overfitting (model.py lines 77-83).
The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

|Type | Input |
|:---------------------:|:---------------------------------------------:| 
|Lambda | 60x60x3|
|Convolution ||60x60x3 |
|Convolution ||28x28x24 |
|Convolution ||11x11x36 |
|Convolution ||3x3x48 |
|Dense ||100 |
|Dense ||50 |
|Dense ||10 |
|Dense ||1 |




#### 3. Creation of the Training Set & Training Process

I used the training data provided by Udacity. I then preprocessed this data by cutting out the upper part of the images. Additionally, I resized it to 60x60 pixels. The left and right images were also used to train the model to recover from the side. The brightness was also randomly changed in order to allow the model to generalize better.
I used an Adam optimizer so that manually training the learning rate wasn't necessary.
