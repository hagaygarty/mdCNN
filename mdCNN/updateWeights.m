%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ net ] = updateWeights(net, ni, momentum , lambda )

batchNum=net.hyperParam.batchNum;

%update network weights
for k=size(net.layers,2):-1:1
    if (net.layers{k}.properties.numWeights==0)
        continue;
    end
    
    if (isequal(net.layers{k}.properties.type,net.types.batchNorm)) % is batchnorm layer
        net.layers{k}.gamma = net.layers{k}.gamma - ni * net.layers{k}.properties.niFactor / batchNum * net.layers{k}.dgamma;
        net.layers{k}.beta  = net.layers{k}.beta  - ni * net.layers{k}.properties.niFactor / batchNum * net.layers{k}.dbeta;
    elseif (isequal(net.layers{k}.properties.type,net.types.fc)) % is fully connected layer
        if (lambda~=0)
            weightDecay = ones(size(net.layers{k}.fcweight));
            weightDecay(1:(end-1),:) = (1-lambda*ni);%do not decay bias
            net.layers{k}.fcweight = weightDecay.*net.layers{k}.fcweight;%weight decay
        end
        if (momentum~=0)
            net.layers{k}.momentum = momentum*net.layers{k}.momentum - ni/batchNum*net.layers{k}.dW;
            net.layers{k}.fcweight = net.layers{k}.fcweight + net.layers{k}.momentum;
        else
            net.layers{k}.fcweight = net.layers{k}.fcweight - ni/batchNum*net.layers{k}.dW;
        end
    else
        for fm=1:net.layers{k}.properties.numFm
            if (momentum~=0)
                net.layers{k}.momentum{fm} = momentum * net.layers{k}.momentum{fm} - ni/batchNum*net.layers{k}.dW{fm};
                net.layers{k}.weight{fm} = (1-lambda*ni)*net.layers{k}.weight{fm} + net.layers{k}.momentum{fm};
            else
                net.layers{k}.weight{fm}  = (1-lambda*ni)*net.layers{k}.weight{fm} - ni/batchNum*net.layers{k}.dW{fm};
              
            end
            
            net.layers{k}.weightFFT{fm} = net.layers{k}.weight{fm};
            for dim=1:net.layers{k}.properties.inputDim
                net.layers{k}.weightFFT{fm} = fft(flip(net.layers{k}.weightFFT{fm},dim),(net.layers{k-1}.properties.sizeFm(dim)+2*net.layers{k}.properties.pad(dim)),dim);
            end
        end
        if (momentum~=0)
            net.layers{k}.momentumBias = momentum * net.layers{k}.momentumBias - ni/batchNum*net.layers{k}.biasdW;
            net.layers{k}.bias = net.layers{k}.bias + net.layers{k}.momentumBias;
        else
            net.layers{k}.bias = net.layers{k}.bias - ni/batchNum*net.layers{k}.biasdW;
        end
    end
end

end

