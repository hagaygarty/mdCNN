# mdCNN
Multidimentioanl CNN. Matlab implementation of convolutional Neural Network framework, supporting 2D and 3D input/kernels

mdCNN is a framework used to train and test deep convulotional network. The framework supports all the major features such as droput, padding, stride, max pooling, L2 regularization, momentum , cross entropy/ MSE 

mdCNN Framework is completely written in matlab and supports 1D , 2D and 3D input size. Unlike most CNN implementations , this framework can process input where every feature map is three dimensional, hence good for 3D image processing such as CT and MRI. Kernels are 3 dimensional and convulotion is done in 3D.


Framework is heavily optimized and quite efficient. While training or testing, all of the CPU cores are participating by using Matlab Built-in Multithreading capabilities.

There are several examples for networks pre-configured to run MNIST , CIFAR10 , and 3dMNIST - a special enhancement of MNIST dataset to 3D volumes.

MNIST Demo reach 99.2% in several minutes, and CIFAR10 demo reaches about 80%

I have used this framework in a project for classifying Vertebras in a 3D CT images. 

=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

To run MNIST demo:
Go into the folder 'Demo/MNIST' , Run 'demoMnist.m' file.
The file will download MNIST dataset and start training the network.

After 15 iterations (several minutes) it will open a GUI where you can test the network performance. 
In addition layer 1 filters will be shown.

=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

Check the 'mdCNN documentation.docx' file for more specification on how to configure a network

For any questions, please contact me at hagaygarty@gmail.com


