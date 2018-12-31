%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ net ] = feedForward(net, input , testTime)
%% feedForward - pass a sample throud the net. Returning an array where the first index is the layer num, second is:
% 1 - the output of each neuron before activation.
% 2 - the output of each neuron after activation.
% 3 - selected dropout matrix
% 4 - indexes of max pooling (the index of the max value in the pooling section)

batchNum=size(input,length(net.layers{1}.properties.sizeOut)+1);

net.layers{1}.outs.activation = input;

for k=2:size(net.layers,2)-1
    input = net.layers{k-1}.outs.activation;
    
    switch net.layers{k}.properties.type
        case net.types.softmax
            %% softmax layer
            maxInput = input;
            for dim=1:length(net.layers{k}.properties.sizeOut)
                maxInput = max(maxInput,[],dim);
            end
            net.layers{k}.outs.expIn = exp(input-repmat(maxInput,net.layers{k}.properties.sizeOut));
            net.layers{k}.outs.sumExp=repmat(sumDim(net.layers{k}.outs.expIn, 1:length(net.layers{k}.properties.sizeOut) ) ,net.layers{k}.properties.sizeOut );
            net.layers{k}.outs.z =net.layers{k}.outs.expIn./net.layers{k}.outs.sumExp;
        case net.types.fc
            %% fully connected layer
            net.layers{k}.outs.z = reshape(net.layers{k}.fcweight.' * [reshape(input, [], batchNum) ; ones(1,batchNum)], [net.layers{k}.properties.sizeOut batchNum]);
        case net.types.reshape
            net.layers{k}.outs.z = reshape(input, [net.layers{k}.properties.sizeOut batchNum]);           
        case net.types.conv
            %% for conv layers
            if ( ~isempty(find(net.layers{k}.properties.pad>0, 1)))
                input = padarray(input, [net.layers{k}.properties.pad 0 0], 0 );
            end

            inputFFT = input;
            for dim=1:net.layers{k}.properties.inputDim
                inputFFT = fft(inputFFT,[],dim);
            end

            Im=cell([net.layers{k}.properties.numFm 1]);
            indexesStride = net.layers{k}.properties.indexesStride;
            i1=indexesStride{1};    i2=indexesStride{2};   i3=indexesStride{3};
            wFFT=net.layers{k}.weightFFT;  bias=net.layers{k}.bias;
            for fm=1:net.layers{k}.properties.numFm
                img = sum(inputFFT.*reshape(repmat(wFFT{fm},[ones(1,ndims(wFFT{fm})) batchNum]),size(inputFFT)),4);

                for dim=1:net.layers{k}.properties.inputDim-1
                    img = ifft(img,[],dim);
                end
                img = ifft(img,[],net.layers{k}.properties.inputDim,'symmetric');
                Im{fm} = bias(fm) + img( i1 , i2 , i3 , : , :);
            end

            net.layers{k}.outs.z = cat(4,Im{:});

            if ( ~isempty(find(net.layers{k}.properties.pooling>1, 1))) %pooling exist
                elemSize = prod(net.layers{k}.properties.pooling);
                szOut=size(net.layers{k}.outs.z);
                poolMat=-1/eps+zeros([elemSize net.layers{k}.properties.numFm*prod(ceil(szOut(1:4)./[net.layers{k}.properties.pooling net.layers{k}.properties.numFm])) batchNum]);

                newIndexes=repmat((0:batchNum-1).',1,length(net.layers{k}.properties.indexes))*numel(net.layers{k}.outs.z)/batchNum + repmat(net.layers{k}.properties.indexes,batchNum,1) ;
                newIndexesReshape=repmat((0:batchNum-1).',1,length(net.layers{k}.properties.indexesReshape))*numel(poolMat)/batchNum + repmat(net.layers{k}.properties.indexesReshape,batchNum,1);
                
                poolMat(newIndexesReshape) = net.layers{k}.outs.z(newIndexes);

                [maxVals, net.layers{k}.outs.maxIdx] = max(poolMat);

                net.layers{k}.outs.z = reshape(maxVals , [net.layers{k}.properties.sizeFm net.layers{k}.properties.numFm batchNum]);
            end
        case net.types.batchNorm
            %% batchNorm layer

                if ( testTime )
                    net.layers{k}.outs.batchMean = net.layers{k}.outs.runningBatchMean;
                    net.layers{k}.outs.batchVar = net.layers{k}.outs.runningBatchVar;
                else
                    net.layers{k}.outs.batchMean = mean(input,length(net.layers{k}.properties.sizeOut)+1);
                    net.layers{k}.outs.batchVar = mean((input-repmat(net.layers{k}.outs.batchMean, [ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]  ) ).^2,length(net.layers{k}.properties.sizeOut)+1) ;
                    if (isempty(net.layers{k}.outs.runningBatchMean))
                        net.layers{k}.outs.runningBatchMean = net.layers{k}.outs.batchMean;
                        net.layers{k}.outs.runningBatchVar  = net.layers{k}.outs.batchVar;
                    else
                        net.layers{k}.outs.runningBatchMean = (1-net.layers{k}.properties.alpha)*net.layers{k}.outs.runningBatchMean + net.layers{k}.properties.alpha*net.layers{k}.outs.batchMean;
                        net.layers{k}.outs.runningBatchVar = (1-net.layers{k}.properties.alpha)*net.layers{k}.outs.runningBatchVar + net.layers{k}.properties.alpha*net.layers{k}.outs.batchVar;
                    end
                    
                end
                net.layers{k}.outs.Xh = (input-repmat(net.layers{k}.outs.batchMean,[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]))./repmat(sqrt(net.layers{k}.properties.EPS+net.layers{k}.outs.batchVar), [ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]);
                net.layers{k}.outs.z = net.layers{k}.outs.Xh.*repmat(net.layers{k}.gamma,[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum])+repmat(net.layers{k}.beta,[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]);
        otherwise
            assert(false,'Error - unknown layer type %s in layer %d\n',net.layers{k}.properties.type,k);
    end
    
    %% do activation + dropout
    if (testTime==1)
        net.layers{k}.outs.activation = net.layers{k}.properties.Activation(net.layers{k}.outs.z*net.layers{k}.properties.dropOut);
    else
        net.layers{k}.outs.activation = net.layers{k}.properties.Activation(net.layers{k}.outs.z);
        if (net.layers{k}.properties.dropOut~=1)
            net.layers{k}.outs.dropout = repmat(binornd(1,net.layers{k}.properties.dropOut,net.layers{k}.properties.sizeOut),[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]); %set dropout matrix
            net.layers{k}.outs.activation = net.layers{k}.outs.activation.*net.layers{k}.outs.dropout;
        end
    end
end

net.layers{end}.outs.activation = net.layers{end-1}.outs.activation; % for loss layer


end

