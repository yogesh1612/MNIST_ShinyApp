---
title: "Implementation of CNN using Keras on MNIST Dataset"

output: html_document
---

## Implementation of CNN using Keras on MNIST Dataset

### Introduction:

The MNIST dataset is most commonly used for the study of image classification. The MNIST database contains images (28 X 28 Pixels) of handwritten digits from 0 to 9 by American Census Bureau employees and American high school students. 


Both Tensorflow and Keras allow us to download the MNIST dataset directly using the API.


### 1-Load the libraries and MNIST Dataset

```{r}
library(keras)
```

Now, load the dataset

```{r}
# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

```

Let's plot any random digit

```{r}
digit <- x_train[200,,]      # select the 5th training image
plot(as.raster(digit, max = 255)) # plot it!
```

It is important to highlight that data is divided into **60,000** training images and **10,000** testing images and each image has size of **28 X 28 pixels**

It means that the shape of x_train is **(60000, 28, 28)** where 60,000 is the number of samples. 

We have to reshape the x_train from 3 dimensions to 4 dimensions as a requirement to process through Keras API.

Further, we have to normalize our data else our computation cost will be very high. We can achieve this by dividing the RGB codes to 255 as follows:

### 2-Data Prepration

```{r}
# Input image dimensions
img_rows <- 28
img_cols <- 28

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255
```

```{r}
cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')
```
The dimensions of our y labels need to match the dimensions of our output layer

```{r}
# Convert class vectors to binary class matrices
num_classes <- 10 # we are classifying numbers from 0-9
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

```


### 3-Convolutional Neural Network Architecture

Keras first creates a new instance of a model object and then add layers to it one after the another. It is called a sequential model API. 

We can add layers to the neural network just by calling model.add and passing in the type of layer we want to add. Finally, we will compile the model with two important information, loss function, and cost optimization algorithm.


Once we execute the above code, Keras will build a TensorFlow model behind the scenes.

Let's first create the model

#### Step 3.1 - Create model


 + The first hidden layer is a convolutional layer called a Convolution2D. We will use 32 filters with        size 3×3 each.
 + Another convolutional layer with 64 filters with size 3×3 each.
 + Then a Max pooling layer with a pool size of 2×2.
 + Then we will use a regularization layer called Dropout. It is configured to randomly exclude 25% of       neurons in the layer in order to reduce overfitting.
 + Then next is a Flatten layer that converts the 2D matrix data to a 1D vector before building the fully   connected layers.
 + After that we will use a fully connected layer with 128 neurons and relu activation function
 + Again, randomly exclude 5% of neurons to reduce overfitting
 + Finally, the output layer which has 10 neurons for the 10 classes and a softmax activation function to    output probability-like predictions for each class.

After deciding the above, we can set up a neural network model with a few lines of code as follows:


```{r}
# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')
```

Now, compile the above model

```{r}
# Compile model
model %>% compile(
          # loss = loss_categorical_crossentropy,
          optimizer = optimizer_adadelta(),
          loss = "categorical_crossentropy",
          metrics = c('accuracy')
          )
```

Once we execute the above code, Keras will build a TensorFlow model behind the scenes.

We can see how many parameters will be learned during the training process by printing model summary as follows:

```{r}
summary(model)
```


#### Step 3.2 - Train the model

First set the epochs and batch size

  + epochs is a hyperparameter that defines the number of times that the learning algorithm will work through the entire training dataset.

  + batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

```{r}
epochs <- 10
batch_size <- 256
```

We can train the model by calling fit method and pass in the training data and the expected output. 

Keras will run the training process and print out the progress to the console. 

When training completes, it will report the final accuracy that was achieved with the training data.

```{r}
# Train model
history <- model %>% fit(
                      x_train, y_train,
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_split = 0.2
                      )
```

We can plot both the training loss and the validation loss at the end of each epoch:


```{r}
plot(history)
```


#### Step 3.3 - Test the model

Now, we will test the performance of model on unseen data by passing in the testing data set and the expected output.

```{r}
scores <- model %>% evaluate(
                  x_test, y_test, verbose = 0
                  )
```

```{r}
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
```
Let's predict categories for the first ten test images and compare them to the actual y values

```{r}
test_sample <- x_test[1:10,,,]
test_sample_array <- array_reshape(test_sample,c(10, img_rows, img_cols, 1))
model%>%predict_classes(test_sample_array)
```
print actual values

```{r}

# note we are not using y_test because we transformed it earlier
mnist$test$y[1:10] 

```


#### Step 3.4 - Save & Load the model

Once we reach the optimum results we can save the model using model.save and pass in the file name. This file will contain everything we need to use our model in another program.

```{r}
model %>% save_model_hdf5("mnist_model.h5")
```

Your model will be saved in the Hierarchical Data Format (HDF) with .h5 extension. It contains multidimensional arrays of scientific data.

We can load our previously trained model by calling the load model function and passing in a file name. 

```{r}
new_model <- load_model_hdf5("mnist_model.h5")
```

Then we call the predict function and pass in the new data for predictions.

```{r}
test_sample <- x_test[1:10,,,]
test_sample_array <- array_reshape(test_sample,c(10, img_rows, img_cols, 1))
new_model%>%predict_classes(test_sample_array)
```


### Summary

 + We learned how to load the MNIST dataset and normalize it.
 + We learned the implementation of CNN using Keras.
 + We saw how to save the trained model and load it later for prediction.
  