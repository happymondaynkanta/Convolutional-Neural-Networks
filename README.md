# Convolutional-Neural-Networks
This repository contains several fundamental implementations of convolutional neural networks 
Convolutional neural networks are a type of neural network that is typically used to process picture, speech, and audio inputs. CNNs are extremely useful for processing pictures and extracting information via convolutional processes. Image classification, object identification, and image segmentation are some of the most common applications (both semantic segmentation and instance segmentation).
Convolutional Neural Network takes advantage of picture structure, resulting in a sparse connection between input and output neurons. In  CNN, each layer conducts convolution where the RGB image is fed into the CNN as an image volume. In general, CNN takes an image as an input and apply a kernel/filter on it to produce the output. CNN also allows parameter sharing between output neurons, which indicates that a feature detector that works well in one region of the image will likely work well in another region of the image.

![1](https://user-images.githubusercontent.com/63404097/150634161-e13c35d0-07b1-4e8c-94ca-65ff3c27bce0.png)


# Basic components of CNN 
## Convolution
A weight matrix, also known as a kernel or a weight matrix, connects each output neuron to a small region in the input. For each convolution layer, we can define numerous kernels, each of which produces an output. A second output is created by moving each filter around the input image. The outputs of each filter are stacked together to form an output volume.

![2](https://user-images.githubusercontent.com/63404097/150634178-a5f74b24-9e59-41c4-beb3-75ab5f8c8d89.png)


![3](https://user-images.githubusercontent.com/63404097/150634146-6a9a9559-d582-459e-89a2-e849eb0ca7ac.png)



## Filters
The convolutional layer's accompanying filter slides through each 3*3 group of pixels from the input. The technique is continued until the filter has covered the entire 3*3 pixel block. Convolving is the term for this type of sliding. The output of the dot product, known as a feature map, stores the results of the dot product.

![4](https://user-images.githubusercontent.com/63404097/150634131-4d58fc7e-fd13-4609-aea6-af736d43f286.png)


## Pooling
After convolution, the feature map we get is sensitive to the placement of the features in the input image. We use pooling to down sample the feature maps to alleviate this sensitivity. As a result, the down sampled feature will be more resistant to changes in the feature's position in the input image ( local translation invariance). We need to conduct a lot of multiplication operations because the input is so large. Fortunately, pooling reduces the amount of computing required during training. There are several types of pooling; the most prevalent options are max pooling and average pooling. We take the maximum value from a tiny region in the feature map while performing max pooling. If we use average pooling, we take the average of all values in the selected region.

![5](https://user-images.githubusercontent.com/63404097/150634096-7b51aa85-6ac2-45b5-bf76-7aa6b65546cd.png)




![6](https://user-images.githubusercontent.com/63404097/150634098-ae228b2b-ea2e-4462-a0a2-7553b61b60c1.png)

