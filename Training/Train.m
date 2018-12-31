%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ net ] = Train( dataset ,  net , numSamplesToTrain )
%% function will train the network on a given dataset

if (~exist('numSamplesToTrain','var'))
    numSamplesToTrain=Inf;
end

logFolder = fullfile(pwd,'Logs');
if ( ~isdir(logFolder) )
    mkdir(logFolder);
end


diary(fullfile(logFolder ,['Console_'  datestr(now,'dd-mm-yyyy_hh-MM-ss') '.txt']));

%dataset = normalize(dataset);

net.runInfoParam.datasetInfo.numTest = length(dataset.I_test);
net.runInfoParam.datasetInfo.numTrain = length(dataset.I);
net.runInfoParam.datasetInfo.firstImSize = num2str(size(dataset.I{1}));
net.runInfoParam.datasetInfo.varFirstIm = var(double(dataset.I{1}(:)));
net.runInfoParam.datasetInfo.minFirstIm = min(double(dataset.I{1}(:)));
net.runInfoParam.datasetInfo.maxFirstIm = max(double(dataset.I{1}(:)));

fprintf('Dataset info - test: %d, train: %d, first sample size:=%s, var=%.2f, min=%f, max=%f\n',...
    net.runInfoParam.datasetInfo.numTest,net.runInfoParam.datasetInfo.numTrain , net.runInfoParam.datasetInfo.firstImSize , ...
    net.runInfoParam.datasetInfo.varFirstIm, net.runInfoParam.datasetInfo.minFirstIm, net.runInfoParam.datasetInfo.maxFirstIm);


%printNetwork(net);

if(net.runInfoParam.verifyBP==1)
    verifyBackProp(net);
    net.runInfoParam.verifyBP=0;% no need to verify anymore
end


assert(net.layers{1}.properties.numFm==1 || net.layers{1}.properties.numFm==size(dataset.I{1},ndims(dataset.I{1})), 'Error - num Fm of input (%d) does not match network configuration (%s)',size(dataset.I{1},ndims(dataset.I{1})),num2str([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm]));
assert(net.layers{end}.properties.numFm>max(dataset.labels) && min(dataset.labels)>=0, ['Error - size of output layer is too small for input. Output layer size is ' num2str(net.layers{end-1}.properties.numFm) ', labels should be in the range 0-' num2str(net.layers{end-1}.properties.numFm-1), '. current labels range is ' num2str(min(dataset.labels)) , '-' num2str(max(dataset.labels))]);
if ( length(unique(dataset.labels)) ~= net.layers{end}.properties.numFm)
    warning(['Training samples does not contain all classes. These should be ' num2str(net.layers{end}.properties.numFm) ' unique classes in training set, but it looks like there are ' num2str(length(unique(dataset.labels))) ' classes']);
end
if ( runstest(dataset.labels) == 1 ) || issorted(dataset.labels) || issorted(fliplr(dataset.labels) )
    warning('Training samples apear not to be in random order. For training to work well, class order in dataset need to be random. Please suffle labels and I (using the same seed) before passing to Train');
end


assert(ndims((zeros([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm])))==ndims(GetNetworkInputs(dataset.I{1},net,1)), 'Error - input does not match network configuration (input size + num FM)');

tic;

if (net.hyperParam.addBackround==1)
    backroundImages = loadBackroundImages();
    fprintf('Finished loading backround images\n');
end


rng(net.runInfoParam.endSeed);

net.runInfoParam.startLoop=clock;
maxSamplesToTrain = numSamplesToTrain + net.runInfoParam.samplesLearned;

fprintf('Start training on %d samples (%.1f epocs, %d batches, batchSize=%d)\n', numSamplesToTrain , numSamplesToTrain/length(dataset.I),  floor(numSamplesToTrain/net.hyperParam.batchNum),net.hyperParam.batchNum);

if (net.runInfoParam.iter==0)
    net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni = net.hyperParam.ni_initial;
else
    net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni = net.runInfoParam.iterInfo(net.runInfoParam.iter).ni;
end

if (~isfield(net.runInfoParam,'loss_train'))
    net.runInfoParam.loss_train=[];
    net.runInfoParam.loss_test=[];
    net.runInfoParam.sucessRate_Test=[];
    net.runInfoParam.sucessRate_Train=[];
end

figure('Name','Training stats');

trainLoopCount = ceil(net.hyperParam.trainLoopCount/net.hyperParam.batchNum)*net.hyperParam.batchNum;
testLoopCount   = ceil(net.hyperParam.testImageNum/net.hyperParam.batchNum)*net.hyperParam.batchNum;

Batch       = zeros([net.layers{1}.properties.sizeOut   net.hyperParam.batchNum]);
expectedOut = zeros([net.layers{end}.properties.sizeOut net.hyperParam.batchNum]);

%% Main epoc loop
while (1)
    net.runInfoParam.iter=net.runInfoParam.iter+1;
    
    fprintf('Iter %-3d| samples=%-4d',net.runInfoParam.iter,net.runInfoParam.samplesLearned+trainLoopCount);
    
    startIter=clock;
    net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=0;
    net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=0;
    rmsErrCnt=0;
    %% start training loop
    batchIdx=0;
    for i=1:trainLoopCount
        batchIdx=batchIdx+1;
        if (net.hyperParam.randomizeTrainingSamples==1)
            idx = randi(length(dataset.I));
        else
            idx = mod(net.runInfoParam.samplesLearned,length(dataset.I))+1;
        end
        sample=double(dataset.I{idx});
        label = dataset.labels(idx);
        
        % do augmentation if rquired in configuration
        if (net.hyperParam.augmentImage==1)
            [sample,complement] = manipulateImage(sample,net.hyperParam.augmentParams.noiseVar , net.hyperParam.augmentParams.maxAngle , net.hyperParam.augmentParams.maxScaleFactor , net.hyperParam.augmentParams.minScaleFactor , net.hyperParam.augmentParams.maxStride, net.hyperParam.augmentParams.maxSigma , net.hyperParam.augmentParams.imageComplement);
        end
        % more optional augmentation, fusing with backround noise 
        if (net.hyperParam.addBackround==1)
            backroundImage=backroundImages{randi(length(backroundImages))};
            starty=randi(1+size(backroundImage,1)-size(sample,1));
            startx=randi(1+size(backroundImage,2)-size(sample,2));
            
            patch = double(backroundImage(starty:(starty+size(sample,1)-1) ,startx:(startx+size(sample,2)-1) ));
            
            switch randi(2)
                case 1
                    sample = imfuse(sample,patch);
                case 2
                    if (complement==1)
                        sample = min(sample,patch);
                    else
                        sample = max(sample,patch);
                    end
            end
            
        end
        
        % add a single sample to the batch
        %GetNetworkInputs will do flipping/scaling to the sample in order to match the network input layer. 
        %Its a helper function not needed if the data is scaled correctly
        Batch(:,:,:,:,batchIdx) = GetNetworkInputs(sample, net, 0); 
        
        expectedOut(:,:,batchIdx)=zeros(net.layers{end}.properties.sizeOut);
        expectedOut(1,label+1,batchIdx)=1;
        
        
        net.runInfoParam.samplesLearned=net.runInfoParam.samplesLearned+1;
        if (batchIdx<net.hyperParam.batchNum)
            continue;
        end
        batchIdx=0;
        
        % train on the batch
        net = backPropagate(net, Batch, expectedOut);

        % Calculate loss
        batchLoss = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
        net.runInfoParam.loss_train(end+1) = mean(batchLoss(:));
        
        % Get classification
        [~,netClassification]  = max(squeeze(net.layers{end}.outs.activation));
        [~,realClassification] = max(squeeze(expectedOut));
        
        net.runInfoParam.sucessRate_Train(end+1) = sum(realClassification==netClassification)/length(realClassification)*100;
        
        net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad+perfomOnNetDerivatives(net,@(x)(rms(x)));
        net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr+rms(net.layers{end}.error(:));
        
        % update weights
        net = updateWeights(net, net.runInfoParam.iterInfo(end).ni, net.hyperParam.momentum , net.hyperParam.lambda);
        
        rmsErrCnt=rmsErrCnt+1;
    end
    endIter=clock;
    
    net.runInfoParam.iterInfo(net.runInfoParam.iter).TrainTime=etime(endIter ,startIter);
    net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr/rmsErrCnt;
    net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad/rmsErrCnt;
    net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights = perfomOnNetWeights(net,@(x)(sqrt(mean(abs(x.^2)))));
    net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights = perfomOnNetWeights(net,@var);
    
    fprintf(' | time=%-5.2f | lossTrain=%f | rmsErr=%f | rmsGrad=%f | meanWeight=%f | varWeight=%f' ,etime(endIter ,startIter), net.runInfoParam.loss_train(end), net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights, net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights );
    
    
    startTesting=clock;
    
    
    %% testSet loop
    batchIdx=0;res=[];lossPerSample=[];
    for i=1:testLoopCount
        batchIdx=batchIdx+1;
        idx=mod(i-1,length(dataset.I_test))+1;
        sample=double(dataset.I_test{idx});
        label = dataset.labels_test(idx);
        
        Batch(:,:,:,:,batchIdx) = GetNetworkInputs(sample, net, 1);
        expectedOut(:,:,batchIdx)=zeros(net.layers{end}.properties.sizeOut);
        expectedOut(1,label+1,batchIdx)=1;
        if (batchIdx<net.hyperParam.batchNum)
            continue;
        end
        batchIdx=0;
        
        %classify the batch from test set
        net = feedForward(net, Batch , 1);
        
        % select the highest probability from network activations in last layer
        [~,netClassification]  = max(squeeze(net.layers{end}.outs.activation));
        [~,realClassification] = max(squeeze(expectedOut)); 
        
        
        res = [res (realClassification==netClassification)];
        
        batchLoss = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
        lossPerSample = [lossPerSample ; squeeze(mean(batchLoss))];
        
        
    end
    endTesting=clock;
    
    net.runInfoParam.loss_test(end+1) = mean(lossPerSample);
    net.runInfoParam.sucessRate_Test(end+1) = sum(res)/length(res)*100;
    
    %% plot training stats
    subplot(2,1,2);
    plot(net.runInfoParam.loss_train); hold on;
    plot(   (1:length(net.runInfoParam.loss_test)) * length(net.runInfoParam.loss_train)/length(net.runInfoParam.loss_test) , net.runInfoParam.loss_test ,'-ok');hold off
    grid on;set(gca, 'YScale', 'log');xlabel('Batch num');ylabel('loss');title('loss');hold off;legend('train set','test set');
    subplot(2,1,1);
    plot(net.runInfoParam.sucessRate_Train); hold on;
    plot(   (1:length(net.runInfoParam.sucessRate_Test)) * length(net.runInfoParam.sucessRate_Train)/length(net.runInfoParam.sucessRate_Test) , net.runInfoParam.sucessRate_Test ,'-ok');hold off
    grid on;xlabel('Batch num');ylabel('success rate %');title('classification');hold off;legend('train set','test set','Location','SE');
    drawnow;
    
    %% save iteration info
    
    net.runInfoParam.endSeed = rng;
    
    if (( net.runInfoParam.loss_test(end) <= net.runInfoParam.minLoss ) || ((exist('net_minM','var')==0)&& (net.runInfoParam.storeMinLossNet==1)))
        if ( net.runInfoParam.loss_test(end) <= net.runInfoParam.minLoss )
            net.runInfoParam.minLoss = net.runInfoParam.loss_test(end);
        end
        if (net.runInfoParam.storeMinLossNet==1)
            net_minM=net; %#ok<NASGU>
        end
    end
    
    if (( net.runInfoParam.sucessRate_Test(end)>=net.runInfoParam.maxsucessRate ) || ((exist('net_maxS','var')==0)&& (net.runInfoParam.storeMinLossNet==1)))
        if ( net.runInfoParam.sucessRate_Test(end)>=net.runInfoParam.maxsucessRate )
            net.runInfoParam.maxsucessRate = net.runInfoParam.sucessRate_Test(end);
        end
        if (net.runInfoParam.storeMinLossNet==1)
            net_maxS=net; %#ok<NASGU>
        end
    end
    
    
    save('net.mat','net');
    
    if (net.runInfoParam.storeMinLossNet==1)
        save('net_maxS.mat','net_maxS');
        save('net_minM.mat','net_minM');
    end
    
    net.runInfoParam.iterInfo(net.runInfoParam.iter).loss=net.runInfoParam.loss_test(end);
    net.runInfoParam.iterInfo(net.runInfoParam.iter).sucessRate_Test=net.runInfoParam.sucessRate_Test(end);
    net.runInfoParam.iterInfo(end+1).ni=net.runInfoParam.iterInfo(end).ni;
    net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime=etime(endTesting  ,startTesting );
    net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime=etime(endTesting  ,net.runInfoParam.startLoop );
    net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt=net.runInfoParam.noImprovementCount;
    
    fprintf(' | lossTest=%f | scesRate=%-5.2f%% | minLoss=%f | maxS=%-5.2f%% | ni=%f' , net.runInfoParam.loss_test(end) , net.runInfoParam.sucessRate_Test(end),net.runInfoParam.minLoss,net.runInfoParam.maxsucessRate,net.runInfoParam.iterInfo(end).ni);
    fprintf(' | tstTime=%.2f',net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime);
    fprintf(' | totalTime=%.2f' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime);
    fprintf(' | noImpCnt=%d/%d' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt, net.hyperParam.noImprovementTh);
    
    fprintf('\n');
    
    if (net.runInfoParam.samplesLearned>=maxSamplesToTrain)
        fprintf('Finish training. max samples reached\n');
        break;
    end
    
    if (( net.runInfoParam.loss_test(end) <= net.runInfoParam.improvementRefLoss ) || ( net.runInfoParam.noImprovementCount > net.hyperParam.noImprovementTh ))
        if (net.runInfoParam.loss_test(end) > net.runInfoParam.improvementRefLoss)
            net.runInfoParam.iterInfo(end).ni=0.5*net.runInfoParam.iterInfo(end).ni;
            fprintf('Updating ni to %f after %d consecutive iterations with no improvement. Ref loss was %f\n', net.runInfoParam.iterInfo(end).ni, net.hyperParam.noImprovementTh, net.runInfoParam.improvementRefLoss);
            net.runInfoParam.improvementRefLoss = Inf;
            if (net.runInfoParam.iterInfo(end).ni< net.hyperParam.ni_final)
                fprintf('Finish testing. ni is smaller then %f\n',net.hyperParam.ni_final);
                break;
            end
        else
            net.runInfoParam.improvementRefLoss = net.runInfoParam.loss_test(end);
        end
        net.runInfoParam.noImprovementCount=0;
    else
        net.runInfoParam.noImprovementCount=net.runInfoParam.noImprovementCount+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
diary off;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ res ] = perfomOnNetWeights( net , func)
    weights=[];
    for k=1:size(net.layers,2)
        if (net.layers{k}.properties.numWeights==0)
            continue
        end
         if (isequal(net.layers{k}.properties.type,net.types.fc)) % is fully connected layer  
             weights=[weights ; net.layers{k}.fcweight(:)]; %#ok<AGROW>
         elseif (isequal(net.layers{k}.properties.type,net.types.conv))
             for fm=1:length(net.layers{k}.weight)
                 weights=[weights ; net.layers{k}.weight{fm}(:)]; %#ok<AGROW>
             end
         elseif (isequal(net.layers{k}.properties.type,net.types.batchNorm))
                 weights=[weights ; net.layers{k}.gamma(:) ; net.layers{k}.beta(:)]; %#ok<AGROW>
         end
    end
    res = func(weights);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ res ] = perfomOnNetDerivatives( net , func)
    dW=[];
    for k=1:size(net.layers,2)
        if (net.layers{k}.properties.numWeights==0)
            continue
        end
         if (isequal(net.layers{k}.properties.type,net.types.fc)) % is fully connected layer  
             dW=[dW ; net.layers{k}.dW(:)]; %#ok<AGROW>
         elseif (isequal(net.layers{k}.properties.type,net.types.conv))
             for fm=1:length(net.layers{k}.weight)
                 dW=[dW ; net.layers{k}.dW{fm}(:)]; %#ok<AGROW>
             end
         elseif (isequal(net.layers{k}.properties.type,net.types.batchNorm))
                 dW=[dW ; net.layers{k}.dgamma(:) ; net.layers{k}.dbeta(:)]; %#ok<AGROW>
         end
    end
    res = func(dW);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ images ] = normalizeSet( images )
    avg = zeros(size(images{1}));
    for idx=1:length(images)
        images{idx} = rgb2ycbcr(double(images{idx}));
        avg = avg+images{idx};
    end    
    avg = avg / length(images);
    for idx=1:length(images)
        images{idx} = images{idx}-avg;
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ images ] = normalize( images )
    images.I = normalizeSet(images.I);
    images.I_test = normalizeSet(images.I_test);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ res ] = rms( x )
    res = sqrt(mean(x.^2));
end