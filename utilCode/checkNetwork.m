function [ confMat4 ] = checkNetwork(net, numSamples, images , useTest)

rnd=0;

if (nargin<3)
    useTest=1;
end

if ( useTest==0 )
    %take from traininng data
    I=images.I;
    labels=images.labels;
else
    %take from testing data
    I=images.I_test;
    labels=images.labels_test;
end

if (numSamples==0 || numSamples==Inf)
    numSamples=length(labels);
end

startIter=clock;
numLabels = net.properties.numOutputs;
labelDist=(zeros(1,numLabels));

range=1:length(labels);

net.hyperParam.testOnMiddlePatchOnly=0;

confMat=zeros(numLabels,numLabels);
confMatTargets=zeros(numLabels,numSamples);
confMatOutputs=zeros(numLabels,numSamples);

fprintf('Testing on %d samples...\n',numSamples);

batchNum = net.hyperParam.batchNum;

BatchSample=zeros([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm 0]); % create an empty batch
expectedOut=zeros([net.layers{end}.properties.sizeOut 0]);% create an empty expected classification
netClassification = zeros(numSamples,1);
realClass         = zeros(numSamples,1);
loss              = zeros(numSamples,1);
datasetIndexes    = zeros(numSamples,1);

for idx=1:numSamples
    if ( rnd==1 )
        rndIndx = randi(length(range));
        imIdx = range(rndIndx);
        range(rndIndx) = [];% in order not to pick the same twice
    else
        imIdx = idx;
    end
    
    if (mod(idx,1000)==0)
        fprintf('Finished checking %d samples (time: %f)\n',idx, etime(clock ,startIter));
        startIter=clock;
    end
    
    datasetIndexes(idx)=imIdx;
    image=I{imIdx};
    label = labels(imIdx);
    labelDist(label+1) = labelDist(label+1) + 1;
    
    batchIdx = 1+mod(idx-1,batchNum);
    BatchSample(:,:,:,:,batchIdx) =  GetNetworkInputs(image, net, 1);
    expectedOut(:,:,batchIdx)=zeros(net.layers{end}.properties.sizeOut);
    expectedOut(1,label+1,end)=1;
    
    if (batchIdx<batchNum)&&(idx<numSamples)
        continue;
    end
    
    % finished filling the batch or reached last sample, classify the batch
    idxRange=idx-size(BatchSample,5)+1:idx;
    
    net = feedForward(net, BatchSample , 1);
    
    
    [~,netClassification(idxRange)]      = max(squeeze(net.layers{end}.outs.activation));
    [~,realClass(idxRange)] = max(squeeze(expectedOut));
    
    cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
    loss(idxRange)              =  squeeze(mean(cost));
    
    confMatOutputs(:,idxRange) = net.layers{end}.outs.activation./repmat(sum(net.layers{end}.outs.activation,2),net.layers{end}.properties.sizeOut);
    
    BatchSample=zeros([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm 0]); % clear the batch
    expectedOut=zeros([net.layers{end}.properties.sizeOut 0]);% create an empty expected classification
    
    
end

for smplIdx=1:length(realClass)
    confMat( netClassification(smplIdx), realClass(smplIdx) ) = confMat( netClassification(smplIdx), realClass(smplIdx) )+1;
    confMatTargets(realClass(smplIdx),smplIdx) = 1;
end


successRate = sum(realClass==netClassification)/length(realClass)*100;
fprintf('success rate %f%%\n',successRate);

failureHist=(zeros(1,numLabels)); % zero till 9

numFailed = sum(realClass~=netClassification);

if (numFailed>0)
    figure('Name',['Selected errors - num nets=' num2str(length(net)) ' success rate ' num2str(successRate) '% , total images ' num2str(length(realClass)) , ' missed ' num2str(numFailed) ' TestSet=' num2str(useTest)]);
end

maxImagePerAxe = 5;
imagePerAxe = min(maxImagePerAxe,round(numFailed^0.5 )+1);

[~, worseLossIdx] = sort(loss(realClass~=netClassification),'descend');
for idx=1:length(worseLossIdx)
    imIdx=datasetIndexes(worseLossIdx(idx));
    label = labels(imIdx);
    failureHist(label+1) = failureHist(label+1)+1;
    
    if ( idx > maxImagePerAxe*maxImagePerAxe)
        continue
    end
    
    image=double(I{imIdx});
    
    net = feedForward(net, GetNetworkInputs(image, net, 1) , 1);
    
    [E, sortedOut] = sort(net.layers{end}.outs.activation); %#ok<ASGLU>
    
    expectedOut=zeros([net.layers{end}.properties.sizeOut 1]);
    expectedOut(1,label+1,1)=1;
    
    cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
    loss  =  mean(cost(:));
    
    h = subplot(imagePerAxe,imagePerAxe,idx);
    axis off
    im = image/max(image(:));
    
    if ( (net.layers{1}.properties.numFm==1)&&(sum(net.layers{1}.properties.sizeFm>1)==3))
        %3d image
        showIso(im,0,0,h);
    else
        imshow(im,'Border','loose');
    end
    title({[ 'Idx=' num2str(imIdx) ' loss=' num2str(loss)],['Ret=' num2str(sortedOut(end)-1) ',' num2str(sortedOut(end-1)-1) ' real=' num2str(label)]}, 'FontSize', 7);
end

figure('Name','Success rate per label');
bar(0:(numLabels-1),100-100*failureHist./labelDist);
xlabel('Label');
ylabel('Success rate %');


figure
plotconfusion(confMatTargets,confMatOutputs);
confMat4=confMat./repmat(sum(confMat,2),1,size(confMat,2));
figure;
surf(confMat4);
ylabel('Label');
xlabel('Network estimation');
zlabel('%');

end

