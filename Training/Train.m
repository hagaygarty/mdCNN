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
 
 
 printNetwork(net);

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
 fprintf('Start training iterations\n');
 
 net.runInfoParam.startLoop=clock;
 maxSamplesToTrain = numSamplesToTrain + net.runInfoParam.samplesLearned;
 
 if (~exist('net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni','var'))
     net.runInfoParam.iterInfo(net.runInfoParam.iter+1).ni = net.hyperParam.ni_initial;
 end
     
         
 %% Main epoc loop
 while (1)
     net.runInfoParam.iter=net.runInfoParam.iter+1;
 
     fprintf('Iter %-3d| Imgs=%-4d',net.runInfoParam.iter,net.runInfoParam.samplesLearned+net.hyperParam.trainLoopCount);
 
     startIter=clock;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=0;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=0;
     rmsErrCnt=0;
     %% start training loop
     BatchSample=zeros([net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm net.hyperParam.batchNum]);
     bIdx=0;
     for i=1:net.hyperParam.trainLoopCount
         bIdx=bIdx+1;
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
         BatchSample(:,:,:,:,bIdx) = GetNetworkInputs(sample, net, 0);
         expectedOut(:,:,bIdx)=zeros(net.layers{end}.properties.sizeOut);
         expectedOut(1,label+1,bIdx)=1;

         
         net.runInfoParam.samplesLearned=net.runInfoParam.samplesLearned+1;
         if (bIdx<net.hyperParam.batchNum)
             continue;
         end
         bIdx=0;
         net = backPropegate(net, BatchSample, expectedOut);
         
         net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad+perfomOnNetDerivatives(net,@(x)(rms(x)));
         net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr=net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr+rms(net.layers{end}.error(:));

         net = updateWeights(net, net.runInfoParam.iterInfo(end).ni, net.hyperParam.momentum , net.hyperParam.lambda);

         rmsErrCnt=rmsErrCnt+1;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%   NULL checking %%%%%%%%%%%%%%%%%%%%%%
         if ((net.hyperParam.testOnNull==1) && (randi(8)==1))
             %combine 4 edges of images , this is learned as null
             th=0.5;
             stride=6;
             pointy = randi(stride)-round(stride/2)+floor(net.layers{1}.properties.sizeFm(1)/2);
             pointx = randi(stride)-round(stride/2)+floor(net.layers{1}.properties.sizeFm(2)/2);
             sample = zeros(size(dataset.I{idx}));
             imgIdxs = randi(length(dataset.I),1,4);
             if (rand(1)>th)
                 sample(1:pointy,1:pointx) = dataset.I{imgIdxs(1)}((end-pointy+1):end,(end-pointx+1):end);
             end
             if (rand(1)>th)
                 sample((pointy+1):end,1:pointx) = dataset.I{imgIdxs(2)}(1:(end-pointy),(end-pointx+1):end);
             end
             if (rand(1)>th)
                 sample(1:(end-pointy),(end-pointx+1):end) = dataset.I{imgIdxs(3)}((pointy+1):end,1:pointx);
             end
             if (rand(1)>th)
                 sample((end-pointy+1):end,(end-pointx+1):end) = dataset.I{imgIdxs(4)}(1:pointy,1:pointx);
             end
             
             
             if (net.hyperParam.augmentImage==1)
                 [sample,complement] = manipulateImage(sample,net.hyperParam.augmentParams.noiseVar, net.hyperParam.augmentParams.maxAngle , net.hyperParam.augmentParams.maxScaleFactor, net.hyperParam.augmentParams.minScaleFactor , net.hyperParam.augmentParams.maxStride, net.hyperParam.augmentParams.maxSigma, net.hyperParam.augmentParams.imageComplement);
             end
 
             if ( (net.hyperParam.addBackround==1) && (randi(2)==1) )
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
 
             sample = GetNetworkInputs(sample, net, 0);
             expectedOut=zeros(1,net.layers{end}.properties.numFm);
             
             net = backPropegate(net, sample, expectedOut);

             if (net.runInfoParam.batchIdx >= net.hyperParam.batchNum)
                net = updateWeights(net, net.runInfoParam.iterInfo(end).ni, net.hyperParam.momentum , net.hyperParam.lambda);
             end
             
         end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%   NULL checking %%%%%%%%%%%%%%%%%%%%%%
         if ((net.hyperParam.testOnNull==1) && (net.hyperParam.addBackround==1) && (randi(8)==1))
             backroundImage=backroundImages{randi(length(backroundImages))};
             starty=randi(1+size(backroundImage,1)-net.layers{1}.properties.sizeFm(1));
             startx=randi(1+size(backroundImage,2)-net.layers{1}.properties.sizeFm(2));
 
             patch = backroundImage(starty:(starty+net.layers{1}.properties.sizeFm(1)-1) ,startx:(startx+net.layers{1}.properties.sizeFm(2)-1) );
             sample = GetNetworkInputs(patch, net, 0);
             expectedOut=zeros(1,net.layers{end}.properties.numFm);

             net = backPropegate(net, sample, expectedOut);
             if (net.runInfoParam.batchIdx >= net.hyperParam.batchNum)
                net = updateWeights(net, net.runInfoParam.iterInfo(end).ni, net.hyperParam.momentum , net.hyperParam.lambda);
             end
             
         end
         
     end
     endIter=clock;
     
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TrainTime=etime(endIter ,startIter);
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr/rmsErrCnt;	 
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad = net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad/rmsErrCnt;	 
     net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights = perfomOnNetWeights(net,@(x)(sqrt(mean(abs(x.^2)))));
     net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights = perfomOnNetWeights(net,@var);
     
     fprintf(' | time=%-5.2f | rmsErr=%f | rmsGrad=%f | meanWeight=%f | varWeight=%f' ,etime(endIter ,startIter), net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsErr, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsGrad, net.runInfoParam.iterInfo(net.runInfoParam.iter).rmsWeights, net.runInfoParam.iterInfo(net.runInfoParam.iter).varWeights );
 
 
     startTesting=clock;
     %% test testing loop
     for i=1:net.hyperParam.testImageNum
         idx=mod(i-1,length(dataset.I_test))+1;
         sample=double(dataset.I_test{idx});
         label = dataset.labels_test(idx);
         
         
         patchAccumRes=0;
         for patchIdx=1:net.hyperParam.testNumPatches
             patch = GetNetworkInputs(sample, net, 1);
             net = feedForward(net, patch , 1);
             patchAccumRes=patchAccumRes+net.layers{end}.outs.activation;
         end

         patchAccumRes = patchAccumRes/net.hyperParam.testNumPatches;
         
         [~,m] = max(patchAccumRes);
         
         expectedOut=zeros([net.layers{end}.properties.sizeOut 1]);
         expectedOut(1,label+1,1)=1;
         
         res(i) = (m-1==label); %#ok<AGROW>
 
         err(i) = sumDim(net.layers{end}.properties.costFunc(patchAccumRes,expectedOut), 1:length(net.layers{end}.properties.sizeOut) );
         
         
     end
     endTesting=clock;
 
     %% save iteration info

     MSE = mean(err);
     sucessRate = sum(res)/length(res)*100;
 
 
     net.runInfoParam.endSeed = rng;
         
     if (( MSE <= net.runInfoParam.minMSE ) || ((exist('net_minM','var')==0)&& (net.runInfoParam.storeMaxMSENet==1)))
		 if ( MSE <= net.runInfoParam.minMSE )
			net.runInfoParam.minMSE = MSE;
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
     
     net.runInfoParam.iterInfo(net.runInfoParam.iter).MSE=MSE;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).sucessRate=sucessRate;
     net.runInfoParam.iterInfo(end+1).ni=net.runInfoParam.iterInfo(end).ni;
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime=etime(endTesting  ,startTesting );
     net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime=etime(endTesting  ,net.runInfoParam.startLoop );
     net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt=net.runInfoParam.noImprovementCount;
     
     fprintf(' | MSE=%f | scesRate=%-5.2f%% | minMSE=%f | maxS=%-5.2f%% | ni=%f' , MSE , sucessRate,net.runInfoParam.minMSE,net.runInfoParam.maxsucessRate,net.runInfoParam.iterInfo(end).ni);
     fprintf(' | tstTime=%.2f',net.runInfoParam.iterInfo(net.runInfoParam.iter).TestTime);
     fprintf(' | totalTime=%.2f' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).TotaolTime);
     fprintf(' | noImpCnt=%d/%d' ,net.runInfoParam.iterInfo(net.runInfoParam.iter).noImpCnt, net.hyperParam.noImprovementTh);

     fprintf('\n');
     
     if (net.runInfoParam.samplesLearned>=maxSamplesToTrain)
         fprintf('Finish training. max samples reached\n');
         break;
     end

     if (( MSE <= net.runInfoParam.improvementRefMSE ) || ( net.runInfoParam.noImprovementCount > net.hyperParam.noImprovementTh ))
         if (MSE > net.runInfoParam.improvementRefMSE)
             net.runInfoParam.iterInfo(end).ni=0.5*net.runInfoParam.iterInfo(end).ni;
             fprintf('Updating ni to %f, Ref was %f\n',net.runInfoParam.iterInfo(end).ni,net.runInfoParam.improvementRefMSE);
             net.runInfoParam.improvementRefMSE = Inf;
             if (net.runInfoParam.iterInfo(end).ni< net.hyperParam.ni_final)
                 fprintf('Finish testing. ni is smaller then %f\n',net.hyperParam.ni_final);
                 break;
             end
         else
             net.runInfoParam.improvementRefMSE = MSE;
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
 
function [ ] = printNetwork( net )
 disp(struct2table(net.hyperParam));
 disp(struct2table(net.runInfoParam));
 
 for k=1:size(net.layers,2)
     fprintf('Layer %d: ',k);
     if (isfield(net.layers{k}.properties,'Activation'))
        fprintf('Activation=%s, dActivation=%s\n', func2str(net.layers{k}.properties.Activation) , func2str(net.layers{k}.properties.dActivation));
     elseif (isfield(net.layers{k}.properties,'lossFunc'))
        fprintf('lossFunc=%s, costFunc=%s\n', func2str(net.layers{k}.properties.lossFunc) , func2str(net.layers{k}.properties.costFunc));
     else
        fprintf('\n');
     end
     disp(struct2table(net.layers{k}.properties));
 end
 
 fprintf('Network properties:\n\n');
 disp(struct2table(net.properties));
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
