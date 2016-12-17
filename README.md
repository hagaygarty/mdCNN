# mdCNN
mdCNN is a MATLAB toolbox implementing Convolutional Neural Networks (CNN) for 2D and 3D inputs. 
Network is Multidimensional, kernels are in 3D and convolution is done in 3D. It is suitable for volumetric inputs such as CT / MRI, but can also support 1D/2D image inputs.

The framework supports all the major features such as droput, padding, stride, max pooling, L2 regularization, momentum, cross entropy/ MSE 
Framework is completely written in matlab and is heavily optimized. While training or testing, all of the CPU cores are participating by using Matlab Built-in Multi-threading.

There are several examples for networks pre-configured to run MNIST, CIFAR10 ,1D CNN, and 3dMNIST - a special enhancement of MNIST dataset to 3D volumes.

MNIST Demo reach 99.2% in several minutes, and CIFAR10 demo reaches about 80%

I have used this framework in a project for classifying Vertebra in a 3D CT images. 

=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

To run MNIST demo:
Go into the folder 'Demo/MNIST' , Run 'demoMnist.m' file.
The file will download MNIST dataset and start training the network.

After 15 iterations (several minutes) it will open a GUI where you can test the network performance. 
In addition layer 1 filters will be shown.

=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

Check the 'mdCNN documentation.docx' file for more specification on how to configure a network

For general questions regarding network design and training , please use the [forum](https://groups.google.com/forum/#!forum/mdcnn-multidimensional-cnn-library-in-matlab).

Any other issues feel free to contact me at hagaygarty@gmail.com 

Hagay



