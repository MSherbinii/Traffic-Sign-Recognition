## Classification of Traffic Signs with LeNet-5 CNN

The purpose of this project is to train an implementation of the LeNet-5 Convolutional Neural
Network for the classification of traffic signs. The model will be used in an application, where
the user can upload a photo of a traffic sign and get the prediction of its class.

## 1. Downloading/Importing Dataset:

The dataset was taken from The German Traffic Sign Recognition Benchmark (GTSRB):
https://benchmark.ini.rub.de/gtsrb_dataset.html. It consists of about 40.000 real colorful
photos of German traffic signs. The images have a (.ppm) extension and their size varies from
15x15 to 250x250 pixels.
![image](https://user-images.githubusercontent.com/47368449/206593556-0fbede7f-b6eb-4090-84f9-47a90c32975d.png)
![image](https://user-images.githubusercontent.com/47368449/206593562-7cd482fc-1d90-48c3-b016-6b3d889c651d.png)

We use the **_wget_** command to download the dataset from an http link, then we extract the zip
file using **_unzip_** command.
![image](https://user-images.githubusercontent.com/47368449/206593577-52060955-f7b2-427e-947b-aef4366c79d1.png)


Now as you can see the file is extracted in the folder **GTSRB.** This folder consists of
![image](https://user-images.githubusercontent.com/47368449/206593596-59d33a97-73dc-436e-b4bf-9754a1de4c25.png)

The subfiles in the Images folder are ranging from 0-41, each is used for a type of traffic sign.


## 2. Libraries

For our Project, we need the following libraries: some standard ones as **_NumPy, OS,_** and
**_Matplotlib_** ; **_cv2_** , a powerful library developed for solving computer vision tasks;
**_sklearn.model_selection.train_test_split_** for splitting the dataset into train and test subsets;
some components from **_tf.keras.models_** and **_tf.keras.layers_** for building the model.
![image](https://user-images.githubusercontent.com/47368449/206593620-8c0ddd97-534f-4352-9a86-a1fb2c415b87.png)


## 3. Read and pre-process image files

We start with reading images from the dataset. The images are distributed over **43** folders
representing **43 classes** as discussed above. We loop through folders and through images, open
them, resize to **32x32 pixels** , convert from **RGB** to **gray** , and save them as **np.arrays**.
![image](https://user-images.githubusercontent.com/47368449/206593634-083ae5e6-ba93-441d-b644-ed2c7bbf0737.png)


We divide images by 255 to get pixel values between 0.0 and 1.0. Finally, we have got a total
amount of 39.209 images assigned to 43 classes.
![image](https://user-images.githubusercontent.com/47368449/206593657-b6f75e14-ccfa-4b07-b5cf-215f1cf78c6b.png)


## 4. Split Dataset into train and test subsets

The dataset has to be split now into train and test subsets. For the test subset, we take out 20%
of the dataset.
![image](https://user-images.githubusercontent.com/47368449/206593675-b57519ea-f77a-4bf2-82a5-6059eb803a17.png)


Now, taking a look into some samples from our dataset. We will pick up 25 random images and
plot them together including their labels.
![image](https://user-images.githubusercontent.com/47368449/206593684-19984f53-d8b3-4cd8-9e80-c9cae40ac299.png)



## 5. Build the model

For our classification task, we will use an implementation of LeNet-5 Convolutional Neural
Network. LeNet-5 was designed by Yann LeCun and others in 1998 and was one of the earliest
convolutional neural networks. Its architecture is extremely simple but very efficient.

There are three Convolutional Layers based on 5x5 filters and followed by average pooling with
2×2 patches. We use the ReLU function for activation as it leads to faster training. Then we add
Dropout Layer with a factor of 0.2 to overcome overfitting. It means that 20% of the input will
be randomly nullified to prevent strong dependencies between layers. We end up with
Flattening and two Dense Layers. In the last Dense Layer, we have to assign the number of
neurons equal to the number of classes, and the Softmax activation function to get probabilities
between 0 and 1. The resulting number of weights in this network is 70,415.
![image](https://user-images.githubusercontent.com/47368449/206593701-b45342c3-c024-4a96-b4aa-710635f90695.png)



## 6. Train the model

As you can see, we have used 50 epoches that gives an opportunity for the Convolution
Network to become more accurate and lower loss.
![image](https://user-images.githubusercontent.com/47368449/206593735-9ef976ec-361b-42d6-8e1c-78c929c53632.png)

![image](https://user-images.githubusercontent.com/47368449/206593709-789be530-f5ac-4a90-b083-10770fed28e0.png)



## 7. Evaluate training results

After 50 epochs we received an accuracy of about 99%, which is quite good. Thus we stop here.
![image](https://user-images.githubusercontent.com/47368449/206593750-941b3870-a373-49e6-8a2a-6b7416c30535.png)


Now, let’s plot loss and accuracy on the training and validation sets.
![image](https://user-images.githubusercontent.com/47368449/206593756-2b0c027b-c8b1-4992-b0f1-72dfabed36c4.png)

As we see from the plots, train and validation accuracy go close, validation loss doesn't go up.
So, the model looks fine, and we do not have much overfitting here.


## 8. Prediction for samples

Let's take a look at some samples and find the wrong classified pictures. We label the images
with prediction and ground truth classes. If prediction equals ground truth, we assign a green
color to the label, otherwise we make it red.
![image](https://user-images.githubusercontent.com/47368449/206593775-57983c58-7c61-4e6a-959c-644ddf3ad4d4.png)
![image](https://user-images.githubusercontent.com/47368449/206593782-8636e9f6-277a-4eed-a003-bd039c4a9eaa.png)



As we can see here, none of the predicted signs were wrong, lets try another set of images to
test it out again.
![image](https://user-images.githubusercontent.com/47368449/206593787-af34810d-4f46-4134-8b2b-b764120865ce.png)


For image number 3865 we can see that the "No Overtaking" sign was misclassified as "Speed
limit ( 7 0km/h)". Obviously, the image of the sign here is very hard to recognize.

**No Overtaking Sign:**
![image](https://user-images.githubusercontent.com/47368449/206593870-46952524-9141-459f-a0ca-d74ef950adeb.png)



## 9. Save the model

In the end, we save the model. It will be used for further predictions in the application for
traffic signs recognition.
![image](https://user-images.githubusercontent.com/47368449/206593891-89604373-389d-4582-bbb5-5b4a860244b9.png)


