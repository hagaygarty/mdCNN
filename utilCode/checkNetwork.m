function [ confMat4 ] = checkNetwork(nets,num, images , useTest , showMiddlePatchOnly)

  if (~exist('showMiddlePatchOnly','var'))
     showMiddlePatchOnly=0;
  end

  
if ( ~iscell(nets) )
    tmp{1}=nets;
    nets=tmp;
end

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

if (num==0 || num==Inf)
    num=length(labels);
end

startIter=clock;
failedCount=0;
numLabels = nets{1}.properties.numOutputs;
labelDist=(zeros(1,numLabels));
failedIndex = [];
failedMSE = [];
range=1:length(labels);

nets{1}.hyperParam.testOnMiddlePatchOnly=0;

if ( ~isfield(nets{1}.hyperParam, 'testNumPatches'))
    numPatchesToTest =1;
else
    numPatchesToTest = nets{1}.hyperParam.testNumPatches;    
end
confMat=zeros(numLabels,numLabels);
confMat2=zeros(numLabels,num);
confMat3=zeros(numLabels,num);
fprintf('Testing on %d images...\n',num);
for idx=1:num;
    if ( rnd==1 )
        rndIndx = randi(length(range)); 
        imIdx = range(rndIndx);
        range(rndIndx) = [];% in order not to pick the same twice
    else
        imIdx = idx;
    end
    
    
    image=I{imIdx};
    label = labels(imIdx);
    labelDist(label+1) = labelDist(label+1) + 1;

    patchAccumRes=0;
    for patchIdx=1:numPatchesToTest
        patch = GetNetworkInputs(image, nets{1}, 1);
        nets{1} = feedForward(nets{1}, patch , 1);
        patchAccumRes=patchAccumRes+nets{1}.layers{end}.outs.activation;
    end

    patchAccumRes = patchAccumRes/numPatchesToTest;

    [M,m] = max(patchAccumRes);

         
    expectedOut=zeros(nets{1}.layers{end}.properties.numFm,1);
    expectedOut(label+1)=1;

    res(idx) = (m-1==label);
    conf(idx) = M*res(idx);
    
    confMat(label+1 , m ) = confMat(label+1 , m )+1;
    confMat2(label+1,idx) = 1;
    confMat3(:,idx) = patchAccumRes/sum(patchAccumRes);

    err(idx) = nets{1}.layers{end}.properties.costFunc(nets{1}.layers{end}.outs.activation,expectedOut);
    
    if ( res(idx)==0)
        failedCount=failedCount+1;
        failedIndex(failedCount)=imIdx;
        failedMSE(failedCount)=err(idx);
    end
    
    if (mod(idx,1000)==0)
        fprintf('Finished checking %d images (time: %f)\n',idx, etime(clock ,startIter));
        startIter=clock;
    end
end

successRate = sum(res)/length(res)*100;
fprintf('success rate %f%%\n',successRate);

failureHist=(zeros(1,numLabels)); % zero till 9

numFailed = length(res)-sum(res);

if (~isempty(failedIndex))
    figure('Name',['Selected errors - num nets=' num2str(length(nets)) ' success rate ' num2str(successRate) '% , total images ' num2str(length(res)) , ' missed ' num2str(numFailed) ' TestSet=' num2str(useTest)]);
end

maxImagePerAxe = 5;
imagePerAxe = min(maxImagePerAxe,round(numFailed^0.5 )+1);    

[~, worseMSEIdx] = sort(failedMSE,'descend');
imgCount=0;
for idx=worseMSEIdx
    imgCount=imgCount+1;
    imIdx=failedIndex(idx);
    label = labels(imIdx);
    failureHist(label+1) = failureHist(label+1)+1;

    if ( imgCount > maxImagePerAxe*maxImagePerAxe)
        continue
    end

    image=double(I{imIdx});
    
    patchAccumRes=0;
    for patchIdx=1:numPatchesToTest
        patch = GetNetworkInputs(image, nets{1}, 1);
        nets{1} = feedForward(nets{1}, patch , 1);
        patchAccumRes=patchAccumRes+nets{1}.layers{end}.outs.activation;
    end

    patchAccumRes = patchAccumRes/numPatchesToTest;

    [E, sortedOut] = sort(patchAccumRes); %#ok<ASGLU>

    expectedOut=zeros(nets{1}.layers{end}.properties.numFm,1);
    expectedOut(label+1)=1;

    MSE =  nets{1}.layers{end}.properties.costFunc(nets{1}.layers{end}.outs.activation,expectedOut);
    

    h = subplot(imagePerAxe,imagePerAxe,imgCount);
    axis off
    im = image/max(image(:));
    
    if ( (nets{1}.layers{1}.properties.numFm==1)&&(sum(nets{1}.layers{1}.properties.sizeFm>1)==3))
        %3d image
        if ( showMiddlePatchOnly==1)
            imshow(im(:,:,round(size(im,3)/2)),'Border','loose');
        else
            showIso(im,0,0,h);
        end
        
    else
        imshow(im,'Border','loose');
    end
    title({[ 'Idx=' num2str(imIdx) ' MSE=' num2str(MSE)],['Ret=' num2str(sortedOut(end)-1) ',' num2str(sortedOut(end-1)-1) ' real=' num2str(label)]}, 'FontSize', 7); 
end

if (~isempty(failedIndex))
    figure('Name','Success rate per label');
    bar(0:(numLabels-1),100-100*failureHist./labelDist);
    xlabel('Label');
    ylabel('Success rate %');

end

figure
plotconfusion(confMat2,confMat3);
confMat4=confMat./repmat(sum(confMat,2),1,size(confMat,2));
figure;
surf(confMat4);
ylabel('Label');
xlabel('Network estimation');
zlabel('%');

end

