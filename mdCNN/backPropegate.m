%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , All rights reserved.
% This file is part of the mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ layers , error , dW] = backPropegate(layers, input, expectedOut , ni, momentum , errMoethod , lambda)
% 3 steps to doing back prop , first is feedForward, second is calculating the
% errors , third is updating the weights.
% The conv in the code below is calculated using FFT. This is much faster

outs = feedForward(layers, input , 0);

%calculate the errors on every layer 
for k=size(layers,2):-1:1
    if ( k == size(layers,2) ) %first layer
        if (errMoethod == 0) %errMoethod = 0  MSE , 1 = cross entroy
            error{k}=layers{k}.properties.dActivation(outs{1,k}).*(outs{2,k}-expectedOut);
        else       
            error{k}= outs{2,k}-expectedOut;%cross entropy
        end
    else  %other layers
         if (layers{k}.properties.type==1) % is fully connected layer
             error{k} = layers{k}.properties.dActivation(outs{1,k}).*(error{k+1}*layers{k+1}.fcweight(1:(end-1),:)');
             if (layers{k}.properties.dropOut~=1)
                 error{k} = outs{3,k}.*error{k};%dropout
             end
         else % is conv layer
            if (layers{k+1}.properties.type==1) % next layer is fully connected layer
                error{k} = reshape(error{k+1}*layers{k+1}.fcweight(1:(end-1),:)',[layers{k}.properties.out layers{k}.properties.numFm]);
            else   % next layer is conv layer

                szNextError=layers{k+1}.properties.sizeFmInPrevLayer+2*layers{k+1}.properties.pad-layers{k+1}.properties.kernel+1;
                szNextKernel=layers{k+1}.properties.kernel;
 
                %expand with stride of error{k+1}
                if ( ~isempty(find(layers{k+1}.properties.stride>1, 1)))
                    nextErrors = zeros([szNextError layers{k+1}.properties.numFm]);
                    nextErrors( (1:layers{k+1}.properties.stride(1):end) , ( 1:layers{k+1}.properties.stride(2):end), ( 1:layers{k+1}.properties.stride(3):end),:) = errBeforePool{k+1};
                else
                    nextErrors = errBeforePool{k+1};
                end
                
                nextErrorFFT = zeros([(szNextError+szNextKernel-1) layers{k+1}.properties.numFm]);
                kernelFFT= zeros([(szNextError+szNextKernel-1) layers{k+1}.properties.numFm layers{k}.properties.numFm]);
                flipMat = layers{k+1}.flipMat;
                for nextFm=1:layers{k+1}.properties.numFm
                	nextErrorFFT(:,:,:,nextFm) = fftn(nextErrors(:,:,:,nextFm),(szNextError+szNextKernel-1));
                    kernelFFT(:,:,:,nextFm,:)=layers{k+1}.weightFFT{nextFm}.*flipMat;
                end
                kernelFFT = conj(kernelFFT);

                error{k} = zeros([(szNextError+szNextKernel-1) layers{k}.properties.numFm]);
                for fm=1:layers{k}.properties.numFm
                   error{k}(:,:,:,fm)= ifftn(sum(nextErrorFFT.*kernelFFT(:,:,:,:,fm),4),'symmetric');                        
                end
                if ( ~isempty(find(layers{k+1}.properties.pad>0, 1)))
                    error{k}= error{k}( (1+layers{k+1}.properties.pad(1)):(end-layers{k+1}.properties.pad(1)) ,  (1+layers{k+1}.properties.pad(2)):(end-layers{k+1}.properties.pad(2)) , (1+layers{k+1}.properties.pad(3)):(end-layers{k+1}.properties.pad(3)) ,:);
                end    
            end
            error{k} = layers{k}.properties.dActivation(outs{1,k}).*error{k};
            if (layers{k}.properties.dropOut~=1)
                error{k} = outs{3,k}.*error{k};%dropout
            end
            
            %expand with pooling
            if ( ~isempty(find(layers{k}.properties.pooling>1, 1))) %pooling exist
                errBeforePool{k} = zeros([floor((layers{k}.properties.sizeFmInPrevLayer+2*layers{k}.properties.pad-layers{k}.properties.kernel)./layers{k}.properties.stride)+1 layers{k}.properties.numFm]);
                errBeforePool{k}(outs{4,k}) = error{k}(:);
            else
                errBeforePool{k} = error{k};
            end
        end
    end
end

%update network weights
for k=size(layers,2):-1:1
    if ( k == 1 ) %last layer
        prevLayerOutput = input;
        prevLayerType = 2;
    else %other layers
        prevLayerOutput = outs{2,k-1};
        prevLayerType = layers{k-1}.properties.type;
    end

    if (layers{k}.properties.type==1) % is fully connected layer
        if (prevLayerType==2) %prev if conv
            prevLayerOutput = reshape(prevLayerOutput, 1,[]);
        end
        
        dW{k} = [prevLayerOutput 1]'*error{k}(:)';

        if (lambda~=0)
            weightDecay = ones(size(layers{k}.fcweight));
            weightDecay(1:(end-1),:) = (1-lambda*ni);%do not decay bias
            layers{k}.fcweight = weightDecay.*layers{k}.fcweight;%weight decay
        end
        if (momentum~=0)
            layers{k}.momentum = momentum*layers{k}.momentum - ni*dW{k};
            layers{k}.fcweight = layers{k}.fcweight + layers{k}.momentum;
        else
            layers{k}.fcweight = layers{k}.fcweight - ni*dW{k};
        end
    else
        if ( ~isempty(find(layers{k}.properties.pad>0, 1)))
            prevLayerOutput = padarray(prevLayerOutput, [layers{k}.properties.pad 0], 0 );
        end
        
        szPrevOutput=layers{k}.properties.sizeFmInPrevLayer+2*layers{k}.properties.pad;
        szOutput=layers{k}.properties.sizeFmInPrevLayer+2*layers{k}.properties.pad-layers{k}.properties.kernel+1;
        
        %flip and expand with stride of error{k}
        kernelFFT = zeros([szOutput layers{k}.properties.numFm]);
        kernelFFT( end+1-(1:layers{k}.properties.stride(1):end) , end+1-( 1:layers{k}.properties.stride(2):end), end+1-( 1:layers{k}.properties.stride(3):end),:) = errBeforePool{k};
         
        prevLayerOutputFFT = prevLayerOutput;
        for dim=1:layers{k}.properties.inputDim
            prevLayerOutputFFT = fft(prevLayerOutputFFT,[],dim);
            kernelFFT = fft(kernelFFT,szPrevOutput(dim),dim);
        end
        
        for fm=1:layers{k}.properties.numFm
            im = prevLayerOutputFFT.*repmat(kernelFFT(:,:,:,fm),1,1,1,layers{k}.properties.numFmInPrevLayer);
            for dim=1:layers{k}.properties.inputDim
                im = ifft(im,[],dim);
            end
           
            im=real(im);
            
            dW{k}{fm} = im( (end-(szPrevOutput(1)-szOutput(1))):end , (end-(szPrevOutput(2)-szOutput(2))):end , (end-(szPrevOutput(3)-szOutput(3))):end , : );

            if (momentum~=0)
                layers{k}.momentum{fm} = momentum * layers{k}.momentum{fm} - ni*dW{k}{fm};
                layers{k}.weight{fm} = (1-lambda*ni)*layers{k}.weight{fm} + layers{k}.momentum{fm};
            else
                layers{k}.weight{fm}  = (1-lambda*ni)*layers{k}.weight{fm} - ni*dW{k}{fm};
            end
            
            layers{k}.weightFFT{fm} = layers{k}.weight{fm};
            for dim=1:layers{k}.properties.inputDim
                layers{k}.weightFFT{fm} = fft(flip(layers{k}.weightFFT{fm},dim),(layers{k}.properties.sizeFmInPrevLayer(dim)+2*layers{k}.properties.pad(dim)),dim);
            end
        end
        if (momentum~=0)
            layers{k}.momentumBias = momentum * layers{k}.momentumBias - ni*1*squeeze(sum(sum(sum(error{k},1),2),3))';
            layers{k}.bias = layers{k}.bias + layers{k}.momentumBias;
        else
            layers{k}.bias = layers{k}.bias - ni*1*squeeze(sum(sum(sum(error{k},1),2),3))';
        end
    end
end

    
end

