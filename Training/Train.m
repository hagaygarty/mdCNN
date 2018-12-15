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
 
  fprintf('Number of samples to test: %d , to train: %d, first sample size:=%s, var=%.10f, min=%f, max=%f\n',...
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
 
 if (net.runInfoParam.displayConvNet==1)
     figure_handle = figure('Name','Network state ');
 end
 
 if (net.hyperParam.addBackround==1)
    backroundImages = loadBackroundImages();
    fprintf('Finished loading backround images\n');
 end
 
 
 rng(net.runInfoParam.endSeed);
 
 net.runInfoParam.startLoop=clock;
 maxSamplesToTrain = numSamplesToTrain + net.runInfoParam.samplesLearned;

 fprintf('Start training on %d samples (%.1f epocs, %d batches, batchSize=%d)\n', maxSamplesToTrain , maxSamplesToTrain/length(dataset.I),  floor(maxSamplesToTrain/net.hyperParam.batchNum),net.hyperParam.batchNum);       
 
 if (~exist('net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni','var'))
     net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni = net.hyperParam.ni_initial;
 end

 if (~exist('net.runInfoParam.MSE_train','var'))
    net.runInfoParam.MSE_train=[];     
 end

 figure('Name','Loss');
 
 %% Main epoc loop
 while (1)
     net.runInfoParam.iter=net.runInfoParam.iter+1;
 
     fprintf('Iter %-3d| samples=%-4d',net.runInfoParam.iter,net.runInfoParam.samplesLearned+net.hyperParam.trainLoopCount);
 
     startIter=clock;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=0;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=0;
     rmsErrCnt=0;
     %% start training loop
     BatchSample=zeros([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm net.hyperParam.batchNum]);
     batchIdx=0;
     for i=1:net.hyperParam.trainLoopCount
         batchIdx=batchIdx+1;
         if (net.hyperParam.randomizeTrainingSamples==1)
            idx = randi(length(dataset.I));
         else
            idx = mod(net.runInfoParam.samplesLearned,length(dataset.I))+1;
         end
         sample=double(dataset.I{idx});
         label = dataset.labels(idx);
             
         if (net.hyperParam.augmentImage==1)
             [sample,complement] = manipulateImage(sample,net.hyperParam.augmentParams.noiseVar , net.hyperParam.augmentParams.maxAngle , net.hyperParam.augmentParams.maxScaleFactor , net.hyperParam.augmentParams.minScaleFactor , net.hyperParam.augmentParams.maxStride, net.hyperParam.augmentParams.maxSigma , net.hyperParam.augmentParams.imageComplement);
         end
         
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
         
         BatchSample(:,:,:,:,batchIdx) = GetNetworkInputs(sample, net, 0);
         expectedOut(:,:,batchIdx)=zeros(net.layers{end}.properties.sizeOut);
         expectedOut(1,label+1,batchIdx)=1;

         
         net.runInfoParam.samplesLearned=net.runInfoParam.samplesLearned+1;
         if (batchIdx<net.hyperParam.batchNum)
             continue;
         end
         batchIdx=0;
         net = backPropegate(net, BatchSample, expectedOut);
         cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
         net.runInfoParam.MSE_train(end+1) = mean(cost(:));
         
         net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad+perfomOnNetDerivatives(net,@(x)(rms(x)));
         net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr+rms(net.layers{end}.error(:));

         net = updateWeights(net, net.runInfoParam.iterInfo(end).ni, net.hyperParam.momentum , net.hyperParam.lambda);

         rmsErrCnt=rmsErrCnt+1;
     end
     endIter=clock;
     
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TrainTime=etime(endIter ,startIter);
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr/rmsErrCnt;	 
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad/rmsErrCnt;	 
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights = perfomOnNetWeights(net,@(x)(sqrt(mean(abs(x.^2)))));
     net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights = perfomOnNetWeights(net,@var);
     
     fprintf(' | time=%-5.2f | MSE=%f | rmsErr=%f | rmsGrad=%f | meanWeight=%f | varWeight=%f' ,etime(endIter ,startIter), net.runInfoParam.MSE_train(end), net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights, net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights );
 
 
     startTesting=clock;
     
     plot(net.runInfoParam.MSE_train);grid on;set(gca, 'YScale', 'log');xlabel('Batch num');ylabel('loss');title('loss on train set');drawnow
     
     %% test testing loop
     batchIdx=0;res=[];mseSample=[];
     for i=1:net.hyperParam.testImageNum
         batchIdx=batchIdx+1;
         idx=mod(i-1,length(dataset.I_test))+1;
         sample=double(dataset.I_test{idx});
         label = dataset.labels_test(idx);
         
         BatchSample(:,:,:,:,batchIdx) = GetNetworkInputs(sample, net, 1);
         expectedOut(:,:,batchIdx)=zeros(net.layers{end}.properties.sizeOut);
         expectedOut(1,label+1,batchIdx)=1;
         if (batchIdx<net.hyperParam.batchNum)
             continue;
         end
         batchIdx=0;
         
         net = feedForward(net, BatchSample , 1);
         
         [~,maxNet]      = max(squeeze(net.layers{end}.outs.activation));
         [~,maxExpected] = max(squeeze(expectedOut));
         
          
         res = [res (maxExpected==maxNet)];
 
         cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
         mseSample = [mseSample ; squeeze(mean(cost))];
         
         
     end
     endTesting=clock;
 
     %% save iteration info

     MSE_test = mean(mseSample);
     sucessRate = sum(res)/length(res)*100;
 
 
     net.runInfoParam.endSeed = rng;
         
     if (( MSE_test <= net.runInfoParam.minMSE ) || ((exist('net_minM','var')==0)&& (net.runInfoParam.storeMaxMSENet==1)))
		 if ( MSE_test <= net.runInfoParam.minMSE )
			net.runInfoParam.minMSE = MSE_test;
		 end
         if (net.runInfoParam.storeMaxMSENet==1)
            net_minM=net; %#ok<NASGU>
         end
     end
 
     if (( sucessRate>=net.runInfoParam.maxsucessRate ) || ((exist('net_maxS','var')==0)&& (net.runInfoParam.storeMaxMSENet==1)))
		 if ( sucessRate>=net.runInfoParam.maxsucessRate )
			net.runInfoParam.maxsucessRate = sucessRate;
		 end	
         if (net.runInfoParam.storeMaxMSENet==1)
            net_maxS=net; %#ok<NASGU>
         end
     end
     
     
     save('net.mat','net');
     
     if (net.runInfoParam.storeMaxMSENet==1)
         save('net_maxS.mat','net_maxS');
         save('net_minM.mat','net_minM');
     end
     
     net.runInfoParam.iterInfo(net.runInfoParam.iter).MSE=MSE_test;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).sucessRate=sucessRate;
     net.runInfoParam.iterInfo(end+1).ni=net.runInfoParam.iterInfo(end).ni;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime=etime(endTesting  ,startTesting );
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime=etime(endTesting  ,net.runInfoParam.startLoop );
     net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt=net.runInfoParam.noImprovementCount;
     
     fprintf(' | MSE_test=%f | scesRate=%-5.2f%% | minMSE=%f | maxS=%-5.2f%% | ni=%f' , MSE_test , sucessRate,net.runInfoParam.minMSE,net.runInfoParam.maxsucessRate,net.runInfoParam.iterInfo(end).ni);
     fprintf(' | tstTime=%.2f',net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime);
     fprintf(' | totalTime=%.2f' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime);
     fprintf(' | noImpCnt=%d/%d' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt, net.hyperParam.noImprovementTh);

     fprintf('\n');
     
     if (net.runInfoParam.samplesLearned>=maxSamplesToTrain)
         fprintf('Finish training. max samples reached\n');
         break;
     end

     if (( MSE_test <= net.runInfoParam.improvementRefMSE ) || ( net.runInfoParam.noImprovementCount > net.hyperParam.noImprovementTh ))
         if (MSE_test > net.runInfoParam.improvementRefMSE)
             net.runInfoParam.iterInfo(end).ni=0.5*net.runInfoParam.iterInfo(end).ni;
             fprintf('Updating ni to %f, Ref was %f\n',net.runInfoParam.iterInfo(end).ni,net.runInfoParam.improvementRefMSE);
             net.runInfoParam.improvementRefMSE = Inf;
             if (net.runInfoParam.iterInfo(end).ni< net.hyperParam.ni_final)
                 fprintf('Finish testing. ni is smaller then %f\n',net.hyperParam.ni_final);
                 break;
             end
         else
             net.runInfoParam.improvementRefMSE = MSE_test;
         end
         net.runInfoParam.noImprovementCount=0;
     else
         net.runInfoParam.noImprovementCount=net.runInfoParam.noImprovementCount+1;
     end
     
     
     if (net.runInfoParam.displayConvNet==1)
         set(figure_handle,'Name',['Network state (net.runInfoParam.iter=' num2str(net.runInfoParam.iter) ,')']);
         displayNetwork(net , dataset.I{1});
     end
     clear empty_script;
     empty_script;
     
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
