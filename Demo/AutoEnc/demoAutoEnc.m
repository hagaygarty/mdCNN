% this demo will train  atoencoder on MNIST dataset
close all;clear all;

addpath('../../Training' , '../../mdCNN' , '../../utilCode' );

net = CreateNet('../../Configs/autoEnc.conf'); 

dataset_folder = 'MNIST_dataset';

MNIST = getMNISTdata(dataset_folder);    

verifyBackProp(net);

batchNum = net.hyperParam.batchNum;


%msefig=figure('Name','MSE per 100 iteration (test set)');
mseBatchfig=figure('Name','MSE per Batch (train set)');
diffs=figure('Name','Left - network input,  Right - network output');
iter=0;  msePerBatch=[]; mse=[]; maxImgsX=8;maxImgsY=8;
ni=net.hyperParam.ni_initial;
maxEpocs=30;
maxIter = round(length(MNIST.I)/batchNum) * maxEpocs;
while( iter<maxIter)
    iter=iter+1;
    if ( mod(iter,2000)==0)
        ni=max(ni/2,net.hyperParam.ni_final);
        fprintf('Reducing ni to %f\n',ni);
    end
    
    batch=[];
    for bIdx=1:batchNum
        batch(:,:,:,:,bIdx) = double(MNIST.I{randi(length(MNIST.I))})/255;%get random batch from train dataset
    end
    expectedOut = batch;
    net = backPropagate(net, batch, expectedOut);
    net = updateWeights(net, ni, net.hyperParam.momentum , net.hyperParam.lambda);
    netOut = net.layers{end}.outs.activation;
    cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
    msePerBatch(end+1) = mean(cost(:));
   
    if ( mod(iter-1,100)==0)%test network every 100 batches
        batch=[];
        for bIdx=1:batchNum
            batch(:,:,:,:,bIdx) = double(MNIST.I_test{randi(length(MNIST.I_test))})/255;%get random batch from test dataset
        end
        expectedOut = batch;
        
        net = feedForward(net, batch , 1);
        netOut = net.layers{end}.outs.activation;
        cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
        mse(end+1) = mean(cost(:));
        fprintf('iter %d/%d, MSE %f\n',iter-1,maxIter,mse(end));
       
      %  set(0,'CurrentFigure',msefig);
      %  plot(mse);grid on;set(gca, 'YScale', 'log');xlabel('Batch num');ylames('MSE');

        set(0,'CurrentFigure',mseBatchfig);
        plot(msePerBatch); hold on;          
        plot(   (1:length(mse)) * length(msePerBatch)/length(mse) , mse ,'-ok');hold off   
        grid on;set(gca, 'YScale', 'log');xlabel('Batch num');ylabel('MSE');title('loss');legend('train set','test set');
        
        set(0,'CurrentFigure',diffs);
       
        for bIdx=1:min(maxImgsX*maxImgsX,batchNum)
            h=subplot(maxImgsX,maxImgsY,bIdx,'replace');
            axis off
            im=squeeze([batch(:,:,:,:,bIdx) netOut(:,:,:,:,bIdx)]);
            imshow(im,'Border','loose') ;
        end
        drawnow;
    end
end
    