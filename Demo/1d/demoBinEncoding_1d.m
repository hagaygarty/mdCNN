
addpath('../../Training' , '../../mdCNN' , '../../utilCode' );
% label is encoded in the sample in binary format using 4
% bits. ie label 7 will be encoded in the middle of the sample [x x x x 0 1 1 1 x x x x]
% network will learn to classify the samples and will eventually 'look'
% only on the interesting bits and ignore other ones

%Crete a dataset
numTest=200;numTrain=10000;
x=randi(16,1,numTrain+numTest)-1; % generate random test set with random numbers 0 to 15

xBin=[mod(x,2) ;mod(floor(x/2),2) ;mod(floor(x/4),2) ;mod(floor(x/8),2)]; % binary encode x values using 4 bits

% 'hide' the 4 bits inside a larger vector padded with random bits and fixed bits
samples = [repmat((1:10)',1,size(xBin,2)) ; xBin ; rand(10,size(xBin,2))/2*mean(xBin(:))]; 

dataset=[];
for idx=1:size(samples,2)
    if (idx>numTrain)
        dataset.I_test{idx-numTrain} = samples(:,idx-numTrain);
        dataset.labels_test(idx-numTrain)=x(idx-numTrain);
    else
    dataset.I{idx} = samples(:,idx);
    dataset.labels(idx)=x(idx);
    end
end

% dataset created , train the network on the data

net = CreateNet('../../Configs/1d_conv.conf'); % small 1d conv net  
%net = CreateNet('../../Configs/1d.conf');  % small 1d fully connected net,will converge faster

net   =  Train(dataset,net, 200000);

checkNetwork(net,Inf,dataset,1);


