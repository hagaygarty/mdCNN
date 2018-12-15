% this demo will get MNIST data automatically and start a training on network specified in 'mnist.conf'

addpath('../../Training' , '../../mdCNN', '../../utilCode');

net = CreateNet('../../Configs/mnist3d.conf'); 

dataset_folder = 'MNIST3d_dataset';

MNIST3d = getMNIST3Ddata(dataset_folder);    

% start training, will train for 120k images 
net   =  Train(MNIST3d,net, length(MNIST3d.I)*2 ); % 2 epocs

% Notes:
% Train will save the network to a file after each iteration. (net.mat) 
% you can call 'Train' again on an existing net, it will continue training where it stopped.

checkNetwork(net,Inf,MNIST3d,1);

showIso(MNIST3d.I{1},0,1);


