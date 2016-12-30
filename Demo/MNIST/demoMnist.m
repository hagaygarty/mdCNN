% this demo will get MNIST data automatically and start a training on network specified in 'mnist.conf'
% it will reach 94% in several minutes, 99.2% in couple of hours

addpath('../../Training' , '../../mdCNN' , '../../utilCode' );

net = CreateNet('../../Configs/mnist.conf'); 

dataset_folder = 'MNIST_dataset';

MNIST = getMNISTdata(dataset_folder);    

% start training, will train for 15k images. Reach about 96.30% in several minutes. 
% In order to reach 99.2% remove the last parameter (15k) and let it train longer.
% It will stop training automatically (once ni reach below thresh)
net   =  Train(MNIST,net, 15000);


checkNetwork(net,Inf,MNIST,1);

%displayMNIST will open a small GUI where you can check the network on MNIST dataset
displayMNIST(net, 'MNIST_dataset'); 

%displayFilters will show the network filters on some input image
displayFilters(net , MNIST.I{1} , num2str(MNIST.labels(1)));

% Notes:
% Train will save the network to a file after each iteration. (net.mat) 
% you can call 'Train' again on an existing net, it will continue training where it stopped.

