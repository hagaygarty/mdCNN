%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , All rights reserved.
% This file is part of the mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ outs ] = feedForward(layers, input , testTime)

outs = cell([4 size(layers,2)]);%pre allocate memory 

%update the layers
for k=1:size(layers,2)
        if (layers{k}.properties.type==1) % is fully connected layer
            outs{1,k} =[reshape(input, 1,[]) 1] * layers{k}.fcweight;
        else
            % for conv layers
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
            
            outs{1,k} = cat(4,Im{:});
            
            if ( ~isempty(find(layers{k}.properties.pooling>1, 1))) %pooling exist
                elemSize = prod(layers{k}.properties.pooling);
                poolMat=-Inf+zeros([elemSize layers{k}.properties.numFm*prod(ceil(size(outs{1,k})./[layers{k}.properties.pooling layers{k}.properties.numFm]))]);
                poolMat(layers{k}.properties.indexesReshape) = outs{1,k}(layers{k}.properties.indexes);

                [maxVals, maxIdx] = max(poolMat);

                outs{4,k} = layers{k}.properties.indexesIncludeOutBounds(maxIdx + layers{k}.properties.offsets); %indexes for fast pooling expansion
                outs{1,k} = reshape(maxVals , [layers{k}.properties.out layers{k}.properties.numFm]);
            end
        end

        if (k==size(layers,2))
            outs{2,k} = layers{k}.properties.Activation(outs{1,k}*layers{k}.properties.dropOut);
        else
            if (testTime==1)
                outs{2,k} = layers{k}.properties.Activation(outs{1,k}*layers{k}.properties.dropOut);
            else
                outs{2,k} = layers{k}.properties.Activation(outs{1,k});
                if (layers{k}.properties.dropOut~=1)
                    outs{3,k} = binornd(1,layers{k}.properties.dropOut,size(outs{1,k})); %set dropout matrix
                    outs{2,k} = outs{2,k}.*outs{3,k};
                end
            end
        end
   
    input = outs{2,k};
end

end

