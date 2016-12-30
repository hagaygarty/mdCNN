%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ outs ] = feedForward(layers, input , testTime)
%% feedForward - pass a sample throud the net. Returning an array where the first index is the layer num, second is:
% 1 - the output of each neuron before activation.
% 2 - the output of each neuron after activation.
% 3 - selected dropout matrix
% 4 - indexes of max pooling (the index of the max value in the pooling section)

outs = cell(size(layers,2),1);

for k=1:size(layers,2)
    if (layers{k}.properties.type==1) 
        %% fully connected layer
        outs{k}.z =[reshape(input, 1,[]) 1] * layers{k}.fcweight;
    else
        %% for conv layers
        if ( ~isempty(find(layers{k}.properties.pad>0, 1)))
            input = padarray(input, [layers{k}.properties.pad 0], 0 );
        end

        inputFFT = input;
        for dim=1:layers{k}.properties.inputDim
            inputFFT = fft(inputFFT,[],dim);
        end

        Im=cell([layers{k}.properties.numFm 1]);
        indexesStride = layers{k}.properties.indexesStride;
        i1=indexesStride{1};    i2=indexesStride{2};   i3=indexesStride{3};
        wFFT=layers{k}.weightFFT;  bias=layers{k}.bias;
        for fm=1:layers{k}.properties.numFm
            img = ifftn(sum(inputFFT.*wFFT{fm},4),'symmetric');
            Im{fm} = bias(fm) + img( i1 , i2 , i3 , :);
        end

        outs{k}.z = cat(4,Im{:});

        if ( ~isempty(find(layers{k}.properties.pooling>1, 1))) %pooling exist
            elemSize = prod(layers{k}.properties.pooling);
            poolMat=-Inf+zeros([elemSize layers{k}.properties.numFm*prod(ceil(size(outs{k}.z)./[layers{k}.properties.pooling layers{k}.properties.numFm]))]);
            poolMat(layers{k}.properties.indexesReshape) = outs{k}.z(layers{k}.properties.indexes);

            [maxVals, maxIdx] = max(poolMat);

            outs{k}.indexes = layers{k}.properties.indexesIncludeOutBounds(maxIdx + layers{k}.properties.offsets); %indexes for fast pooling expansion
            outs{k}.z = reshape(maxVals , [layers{k}.properties.out layers{k}.properties.numFm]);
        end
    end
    %% do activation + dropout
    if (k==size(layers,2))
        outs{k}.activation = layers{k}.properties.Activation(outs{k}.z*layers{k}.properties.dropOut);
    else
        if (testTime==1)
            outs{k}.activation = layers{k}.properties.Activation(outs{k}.z*layers{k}.properties.dropOut);
        else
            outs{k}.activation = layers{k}.properties.Activation(outs{k}.z);
            if (layers{k}.properties.dropOut~=1)
                outs{k}.dropout = binornd(1,layers{k}.properties.dropOut,size(outs{k}.z)); %set dropout matrix
                outs{k}.activation = outs{k}.activation.*outs{k}.dropout;
            end
        end
    end
   
    input = outs{k}.activation;
end

end

