%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [net] = backPropagate(net, input, expectedOut)
% 3 steps to doing back prop , first is feedForward, second is calculating the
% errors , third is calculating the derivitives
% The conv in the code below is calculated using FFT. This is much faster

net = feedForward(net, input , 0);

batchNum=size(input,length(net.layers{1}.properties.sizeOut)+1);
assert(isequal(size(expectedOut) , size(net.layers{end}.outs.activation)));
net.layers{end}.error = net.layers{end}.properties.lossFunc(net.layers{end}.outs.activation,expectedOut);
%% calculate the errors on every layer
for k=size(net.layers,2)-1:-1:2
    % reshape the error to match the layer out type
    net.layers{k}.error = net.layers{k+1}.error;
    
    
    %pass through activation
    net.layers{k}.error =net.layers{k}.error.*net.layers{k}.properties.dActivation(net.layers{k}.outs.z);
    if (net.layers{k}.properties.dropOut~=1)
        net.layers{k}.error = net.layers{k}.outs.dropout.*net.layers{k}.error;%dropout
    end
    
    %calc dW
    switch net.layers{k}.properties.type
        case net.types.softmax
            net.layers{k}.error =(net.layers{k}.outs.sumExp.*net.layers{k}.error- repmat(sumDim(net.layers{k}.outs.expIn.*net.layers{k}.error, 1:length(net.layers{k}.properties.sizeOut)),net.layers{k}.properties.sizeOut)  ).*net.layers{k}.outs.expIn./net.layers{k}.outs.sumExp.^2;
        case net.types.fc
            net.layers{k}.dW = [reshape(net.layers{k-1}.outs.activation, [], batchNum) ; ones(1,batchNum) ]*reshape(net.layers{k}.error, [], batchNum).' ;
            net.layers{k}.error = net.layers{k}.fcweight(1:(end-1),:)*reshape(net.layers{k}.error, [], batchNum);
            net.layers{k}.error = reshape(net.layers{k}.error,[net.layers{k-1}.properties.sizeOut batchNum]);
        case net.types.batchNorm
            N = numel(net.layers{k+1}.error)/numel(net.layers{k}.outs.batchMean);
            net.layers{k}.dbeta  = sum(net.layers{k}.error,length(net.layers{k}.properties.sizeOut)+1);
            net.layers{k}.dgamma = sum(net.layers{k}.error.*net.layers{k}.outs.Xh,length(net.layers{k}.properties.sizeOut)+1);
            net.layers{k}.error  = repmat(net.layers{k}.gamma./sqrt(net.layers{k}.properties.EPS+net.layers{k}.outs.batchVar)/N , [ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]).* (N*reshape(net.layers{k}.error,size(net.layers{k}.outs.Xh))- repmat(net.layers{k}.dgamma,[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]).*net.layers{k}.outs.Xh - repmat(net.layers{k}.dbeta,[ones(1,length(net.layers{k}.properties.sizeOut)) batchNum]) );
        case net.types.reshape
            net.layers{k}.error  = reshape(net.layers{k}.error,[net.layers{k-1}.properties.sizeOut batchNum]);
        case net.types.conv
            %expand with pooling
            if ( ~isempty(find(net.layers{k}.properties.pooling>1, 1))) %pooling exist
                nextErrors = zeros([floor((net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1 net.layers{k}.properties.numFm batchNum]);
                indexes = reshape(repmat((0:batchNum-1),length(net.layers{k}.properties.offsets),1),1,length(net.layers{k}.properties.offsets),batchNum)*numel(nextErrors)/batchNum + (net.layers{k}.properties.indexesIncludeOutBounds(net.layers{k}.outs.maxIdx + repmat(net.layers{k}.properties.offsets,1,1,batchNum) )); %indexes for fast pooling expansion
                nextErrors(indexes) = net.layers{k}.error;
                net.layers{k}.error = nextErrors;
            end
            
            % update weights
            net.layers{k}.biasdW = squeeze(sum(sum(sum(sum(net.layers{k}.error,1),2),3),5));
            
            prevLayerOutput = net.layers{k-1}.outs.activation;
            if ( ~isempty(find(net.layers{k}.properties.pad>0, 1)))
                prevLayerOutput = padarray(prevLayerOutput, [net.layers{k}.properties.pad 0 0], 0 );
            end
            
            szPrevOutput=net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad;
            szOutput=net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel+1;
            
            %flip and expand with stride of net.layers{k}.error
            kernelFFT = zeros([szOutput net.layers{k}.properties.numFm batchNum]);
            kernelFFT( end+1-(1:net.layers{k}.properties.stride(1):end) , end+1-( 1:net.layers{k}.properties.stride(2):end), end+1-( 1:net.layers{k}.properties.stride(3):end),:,:) = net.layers{k}.error;
            
            prevLayerOutputFFT = prevLayerOutput;
            for dim=1:net.layers{k}.properties.inputDim
                prevLayerOutputFFT = fft(prevLayerOutputFFT,[],dim);
                kernelFFT = fft(kernelFFT,szPrevOutput(dim),dim);
            end
            
            for fm=1:net.layers{k}.properties.numFm
                im = prevLayerOutputFFT.*repmat(kernelFFT(:,:,:,fm,:),1,1,1,net.layers{k-1}.properties.numFm);
                for dim=1:net.layers{k}.properties.inputDim
                    im = ifft(im,[],dim);
                end
                im=real(im);
                net.layers{k}.dW{fm} = sum ( im( (end-(szPrevOutput(1)-szOutput(1))):end , (end-(szPrevOutput(2)-szOutput(2))):end , (end-(szPrevOutput(3)-szOutput(3))):end , : ,:), 5);
            end
            
            if (net.properties.skipLastLayerErrorCalc==0) || (k>2) % to save cycles no need to propogate to input layer
                szNextError=net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel+1;
                szNextKernel=net.layers{k}.properties.kernel;
                
                
                %expand with stride of net.layers{k}.error
                if ( ~isempty(find(net.layers{k}.properties.stride>1, 1)))
                    nextErrors = zeros([szNextError net.layers{k}.properties.numFm batchNum]);
                    nextErrors( (1:net.layers{k}.properties.stride(1):end) , ( 1:net.layers{k}.properties.stride(2):end), ( 1:net.layers{k}.properties.stride(3):end),:,:) = net.layers{k}.error;
                    net.layers{k}.error=nextErrors;
                end
                
                nextErrorFFT = zeros([(szNextError+szNextKernel-1) net.layers{k}.properties.numFm batchNum]);
                kernelFFT= zeros([(szNextError+szNextKernel-1) net.layers{k}.properties.numFm net.layers{k-1}.properties.numFm]);
                flipMat = net.layers{k}.flipMat;
                for nextFm=1:net.layers{k}.properties.numFm
                    sz=size(net.layers{k}.error); sz = (szNextError+szNextKernel-1) - sz(1:length(szNextError+szNextKernel-1));
                    im=padarray(net.layers{k}.error(:,:,:,nextFm,:),sz,'post');
                    for dim=1:net.layers{k}.properties.inputDim
                        im = fft(im,[],dim);
                    end
                    nextErrorFFT(:,:,:,nextFm,:) = im;
                    kernelFFT(:,:,:,nextFm,:)=net.layers{k}.weightFFT{nextFm}.*flipMat;
                end
                kernelFFT = conj(kernelFFT);
                
                net.layers{k}.error = zeros([(szNextError+szNextKernel-1) net.layers{k-1}.properties.numFm batchNum]);
                
                for fm=1:net.layers{k-1}.properties.numFm
                    Im=sum(nextErrorFFT.*repmat(kernelFFT(:,:,:,:,fm),[ones(1,ndims(nextErrorFFT)-1) batchNum]),4);
                    for dim=1:net.layers{k}.properties.inputDim-1
                        Im = ifft(Im,[],dim);
                    end
                    Im = ifft(Im,[],net.layers{k}.properties.inputDim,'symmetric');
                    net.layers{k}.error(:,:,:,fm,:)= Im;
                end
                if ( ~isempty(find(net.layers{k}.properties.pad>0, 1)))
                    net.layers{k}.error= net.layers{k}.error( (1+net.layers{k}.properties.pad(1)):(end-net.layers{k}.properties.pad(1)) ,  (1+net.layers{k}.properties.pad(2)):(end-net.layers{k}.properties.pad(2)) , (1+net.layers{k}.properties.pad(3)):(end-net.layers{k}.properties.pad(3)) ,:,:);
                end
            end
    end
end

end

