% this will get CIFAR10 data automatically and start a training on a default
% net configuration 'cifar10.conf'

addpath('../../Training' , '../../mdCNN', '../../utilCode' );

% create network from config file
net = CreateNet('../../Configs/cifar10.conf'); 

dataset_folder = 'CIFAR10_dataset';

% load MNIST data
CIFAR10 = getCIFAR10data(dataset_folder);    

% start trainninig
net   =  Train(CIFAR10,net); 

checkNetwork(net,Inf,CIFAR10,1);

displayCIFAR10(net, fullfile(dataset_folder ,'CIFAR10.mat'),1);

