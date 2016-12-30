%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [net] = backPropegate(net, input, expectedOut)
% 3 steps to doing back prop , first is feedForward, second is calculating the
% errors , third is calculating the derivitives
% The conv in the code below is calculated using FFT. This is much faster

outs = feedForward(net.layers, input , 0);

%% calculate the errors on every layer 
for k=size(net.layers,2):-1:1
    if ( k == size(net.layers,2) ) %first layer
        if (net.hyperParam.errorMethod == 0) %errMoethod = 0  MSE , 1 = cross entroy
            net.layers{k}.error = net.layers{k}.properties.dActivation(outs{k}.z).*(outs{k}.activation-expectedOut);
        else       
            net.layers{k}.error = outs{k}.activation-expectedOut;%cross entropy
        end
    else  %other layers
         if (net.layers{k}.properties.type==1) % is fully connected layer
             net.layers{k}.error = net.layers{k}.properties.dActivation(outs{k}.z).*(net.layers{k+1}.error*net.layers{k+1}.fcweight(1:(end-1),:)');
             if (net.layers{k}.properties.dropOut~=1)
                 net.layers{k}.error = outs{k}.dropout.*net.layers{k}.error;%dropout
             end
         else % is conv layer
            if (net.layers{k+1}.properties.type==1) % next layer is fully connected layer
                net.layers{k}.error = reshape(net.layers{k+1}.error*net.layers{k+1}.fcweight(1:(end-1),:)',[net.layers{k}.properties.out net.layers{k}.properties.numFm]);
            else   % next layer is conv layer

                szNextError=net.layers{k+1}.properties.sizeFmInPrevLayer+2*net.layers{k+1}.properties.pad-net.layers{k+1}.properties.kernel+1;
                szNextKernel=net.layers{k+1}.properties.kernel;
 
                %expand with stride of net.layers{k+1}.error
                if ( ~isempty(find(net.layers{k+1}.properties.stride>1, 1)))
                    nextErrors = zeros([szNextError net.layers{k+1}.properties.numFm]);
                    nextErrors( (1:net.layers{k+1}.properties.stride(1):end) , ( 1:net.layers{k+1}.properties.stride(2):end), ( 1:net.layers{k+1}.properties.stride(3):end),:) = errBeforePool{k+1};
                else
                    nextErrors = errBeforePool{k+1};
                end
                
                nextErrorFFT = zeros([(szNextError+szNextKernel-1) net.layers{k+1}.properties.numFm]);
                kernelFFT= zeros([(szNextError+szNextKernel-1) net.layers{k+1}.properties.numFm net.layers{k}.properties.numFm]);
                flipMat = net.layers{k+1}.flipMat;
                for nextFm=1:net.layers{k+1}.properties.numFm
                	nextErrorFFT(:,:,:,nextFm) = fftn(nextErrors(:,:,:,nextFm),(szNextError+szNextKernel-1));
                    kernelFFT(:,:,:,nextFm,:)=net.layers{k+1}.weightFFT{nextFm}.*flipMat;
                end
                kernelFFT = conj(kernelFFT);

                net.layers{k}.error = zeros([(szNextError+szNextKernel-1) net.layers{k}.properties.numFm]);
                for fm=1:net.layers{k}.properties.numFm
                   net.layers{k}.error(:,:,:,fm)= ifftn(sum(nextErrorFFT.*kernelFFT(:,:,:,:,fm),4),'symmetric');                        
                end
                if ( ~isempty(find(net.layers{k+1}.properties.pad>0, 1)))
                    net.layers{k}.error= net.layers{k}.error( (1+net.layers{k+1}.properties.pad(1)):(end-net.layers{k+1}.properties.pad(1)) ,  (1+net.layers{k+1}.properties.pad(2)):(end-net.layers{k+1}.properties.pad(2)) , (1+net.layers{k+1}.properties.pad(3)):(end-net.layers{k+1}.properties.pad(3)) ,:);
                end    
            end
            net.layers{k}.error = net.layers{k}.properties.dActivation(outs{k}.z).*net.layers{k}.error;
            if (net.layers{k}.properties.dropOut~=1)
                net.layers{k}.error = outs{k}.dropout.*net.layers{k}.error;%dropout
            end
            
            %expand with pooling
            if ( ~isempty(find(net.layers{k}.properties.pooling>1, 1))) %pooling exist
                errBeforePool{k} = zeros([floor((net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1 net.layers{k}.properties.numFm]);
                errBeforePool{k}(outs{k}.indexes) = net.layers{k}.error(:);
            else
                errBeforePool{k} = net.layers{k}.error;
            end
            
            net.layers{k}.biasdW = net.layers{k}.biasdW + squeeze(sum(sum(sum(net.layers{k}.error,1),2),3))';

        end
    end
end

%% accumulate the derivitives

for k=size(net.layers,2):-1:1
    if ( k == 1 ) %last layer
        prevLayerOutput = input;
        prevLayerType = 2;
    else %other layers
        prevLayerOutput = outs{k-1}.activation;
        prevLayerType = net.layers{k-1}.properties.type;
    end

    if (net.layers{k}.properties.type==1) % is fully connected layer
        if (prevLayerType==2) %prev if conv
            prevLayerOutput = reshape(prevLayerOutput, 1,[]);
        end
        
        net.layers{k}.dW = net.layers{k}.dW + [prevLayerOutput 1]'*net.layers{k}.error(:)';
    else
        if ( ~isempty(find(net.layers{k}.properties.pad>0, 1)))
            prevLayerOutput = padarray(prevLayerOutput, [net.layers{k}.properties.pad 0], 0 );
        end
        
        szPrevOutput=net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad;
        szOutput=net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel+1;
        
        %flip and expand with stride of net.layers{k}.error
        kernelFFT = zeros([szOutput net.layers{k}.properties.numFm]);
        kernelFFT( end+1-(1:net.layers{k}.properties.stride(1):end) , end+1-( 1:net.layers{k}.properties.stride(2):end), end+1-( 1:net.layers{k}.properties.stride(3):end),:) = errBeforePool{k};
         
        prevLayerOutputFFT = prevLayerOutput;
        for dim=1:net.layers{k}.properties.inputDim
            prevLayerOutputFFT = fft(prevLayerOutputFFT,[],dim);
            kernelFFT = fft(kernelFFT,szPrevOutput(dim),dim);
        end
        
        for fm=1:net.layers{k}.properties.numFm
            im = prevLayerOutputFFT.*repmat(kernelFFT(:,:,:,fm),1,1,1,net.layers{k}.properties.numFmInPrevLayer);
            for dim=1:net.layers{k}.properties.inputDim
                im = ifft(im,[],dim);
            end
            im=real(im);
            net.layers{k}.dW{fm} = net.layers{k}.dW{fm} + im( (end-(szPrevOutput(1)-szOutput(1))):end , (end-(szPrevOutput(2)-szOutput(2))):end , (end-(szPrevOutput(3)-szOutput(3))):end , : );
        end
    end
end

net.runInfoParam.batchIdx = net.runInfoParam.batchIdx + 1;
    
end

