% this demo will get MNIST data automatically and start a training on network specified in 'mnist.conf'

addpath('../../Training' , '../../mdCNN' , '../../utilCode' );

net = CreateNet('../../Configs/mnist.conf'); 

dataset_folder = 'MNIST_dataset';

MNIST = getMNISTdata(dataset_folder);    

% start training, will train for 180k images. Reach about 98.7% in several minutes. 
% In order to reach better accuracy, remove the last parameter and let it train longer.
% It will stop training automatically (once ni reach below thresh)
net   =  Train(MNIST,net, length(MNIST.I)*3 ); % 3 epocs 


checkNetwork(net,Inf,MNIST,1);

%displayMNIST will open a small GUI where you can check the network on MNIST dataset
displayMNIST(net, 'MNIST_dataset'); 

%displayFilters will show the network filters on some input image
displayFilters(net , MNIST.I{1} , num2str(MNIST.labels(1)));

% Notes:
% Train will save the network to a file after each iteration. (net.mat) 
% you can call 'Train' again on an existing net, it will continue training where it stopped.

