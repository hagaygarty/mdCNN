%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ net ] = initNetWeight( net )
rng(0);
fprintf('multi dimentional CNN , Hagay Garty 2016 | hagaygarty@gmail.com\nInitializing network..\n');

%init W
prevLayerActivation=1; %for input
net.properties.numWeights = 0;
assert( isequal(net.layers{1}.properties.type,net.types.input), 'Error - first layer must be input layer (type =-2)');
assert( isequal(net.layers{1}.properties.sizeFm(net.layers{1}.properties.sizeFm>0),net.layers{1}.properties.sizeFm), 'Error - sizeFm cannot have dim 0');
assert( ((length(net.layers{1}.properties.sizeFm)==1) || (isequal(net.layers{1}.properties.sizeFm(net.layers{1}.properties.sizeFm>1),net.layers{1}.properties.sizeFm))), 'Error - sizeFm cannot have useless 1 dims');
assert( isfield(net.layers{1}.properties,'numFm'), 'Error - numFm is not defined in first layer. Example: for 32x32 rgb please set to numFm=3');
assert( isempty(find(net.layers{1}.properties.sizeFm<1, 1)), 'Error - sizeFm must be >=1 for all dimensions');
assert( ((~isempty(net.layers{1}.properties.sizeFm)) && (length(net.layers{1}.properties.sizeFm)<=3)), 'Error - num of dimensions must be >0 and <=3');

assert( isfield(net.layers{1}.properties,'sizeFm'), 'Error - sizeFm is not defined in hyperParam. Example: for 32x32 rgb please set to [32 32] , numFm=3');
assert( length(net.layers{1}.properties.sizeFm)<=3, 'Error - sizeFm cannot have more then 3 dimensions');


assert( isequal(net.layers{end}.properties.type,net.types.output), 'Error - last layer must be output layer');

net.layers{1}.properties.sizeFm = [net.layers{1}.properties.sizeFm 1 1 1];
net.layers{1}.properties.sizeFm = net.layers{1}.properties.sizeFm(1:3);
net.layers{1}.properties.numWeights = 0;



if (isscalar(net.hyperParam.augmentParams.maxStride))
    net.hyperParam.augmentParams.maxStride = net.hyperParam.augmentParams.maxStride*ones(1,length(net.layers{1}.properties.sizeFm)).*(net.layers{1}.properties.sizeFm>1);
end

for k=1:size(net.layers,2)
    assert( isfield(net.layers{k}.properties,'type')==1, 'Error - missing type definition in layer %d',k);
    fprintf('Initializing layer %d - %s\n',k,net.layers{k}.properties.type);
    switch net.layers{k}.properties.type
        case net.types.input
            assert(k==1,'Error input layer must be the first layer (%d)\n',k);
            if (isfield(net.layers{k}.properties,'Activation')==0)
                net.layers{k}.properties.Activation=@Unit;
            end
            if (isfield(net.layers{k}.properties,'dActivation')==0)
                net.layers{k}.properties.dActivation=@dUnit;
            end
            net.layers{k}.properties.sizeOut = [net.layers{k}.properties.sizeFm net.layers{k}.properties.numFm];
        case net.types.softmax
            net.layers{k}.properties.numFm = net.layers{k-1}.properties.numFm;
            if (isfield(net.layers{k}.properties,'Activation')==0)
                net.layers{k}.properties.Activation=@Unit;
            end
            if (isfield(net.layers{k}.properties,'dActivation')==0)
                net.layers{k}.properties.dActivation=@dUnit;
            end
        case net.types.fc
        case net.types.reshape
            if (isfield(net.layers{k}.properties,'Activation')==0)
                net.layers{k}.properties.Activation=@Unit;
            end
            if (isfield(net.layers{k}.properties,'dActivation')==0)
                net.layers{k}.properties.dActivation=@dUnit;
            end
        case net.types.conv
        case net.types.batchNorm
            assert( isfield(net.layers{k}.properties,'numFm')==0, 'Error - no need to specify numFm in batchnorm layer, its inherited from previous layer. in layer %d',k);
            assert( net.hyperParam.batchNum>=2, 'Error - cannot use batch norm layer if batchSize<2. in layer %d',k);
            net.layers{k}.properties.numFm = net.layers{k-1}.properties.numFm;
            if (isfield(net.layers{k}.properties,'EPS')==0)
                net.layers{k}.properties.EPS=1e-5;
            end            
            if (isfield(net.layers{k}.properties,'niFactor')==0)
                net.layers{k}.properties.niFactor=1;
            end            
            if (isfield(net.layers{k}.properties,'Activation')==0)
                net.layers{k}.properties.Activation=@Unit;
            end
            if (isfield(net.layers{k}.properties,'dActivation')==0)
                net.layers{k}.properties.dActivation=@dUnit;
            end
            if (isfield(net.layers{k}.properties,'initGamma')==0)
                net.layers{k}.properties.initGamma = 1;
            end
            net.layers{k}.gamma=net.layers{k}.properties.initGamma * ones([net.layers{k-1}.properties.sizeFm net.layers{k-1}.properties.numFm]);
            
            if (isfield(net.layers{k}.properties,'initBeta')==0)
                net.layers{k}.properties.initBeta = 0;
            end
            net.layers{k}.beta=net.layers{k}.properties.initBeta * ones([net.layers{k-1}.properties.sizeFm net.layers{k-1}.properties.numFm]);

            net.layers{k}.properties.numWeights = numel(net.layers{k}.gamma)+numel(net.layers{k}.beta);
            
            if (isfield(net.layers{k}.properties,'alpha')==0)
                net.layers{k}.properties.alpha=2^-5;
            end
            assert( (net.layers{k}.properties.alpha<=1)&&(net.layers{k}.properties.alpha>=0),'alpha must be in the range [0 .. 1], layer %d\n',k);
            net.layers{k}.outs.runningBatchMean = [];
            net.layers{k}.outs.runningBatchVar = [];
        case net.types.output
            assert(k==size(net.layers,2),'Error - output layer must be the last layer, layer (%d)\n',k);
            net.layers{k}.properties.sizeFm = net.layers{k-1}.properties.sizeFm;
            net.layers{k}.properties.numFm = net.layers{k-1}.properties.numFm;
            net.layers{k}.properties.sizeOut = [net.layers{k}.properties.sizeFm net.layers{k}.properties.numFm];
            net.layers{k}.properties.Activation=@Unit;
            net.layers{k}.properties.dActivation=@dUnit;
            continue;
        otherwise
            assert(false,'Error - unknown layer type %s in layer %d\n',net.layers{k}.properties.type,k);
    end
    
    assert( isfield(net.layers{k}.properties,'numFm')==1, 'Error - missing numFM definition in layer %d',k);
    
    
    if (isfield(net.layers{k}.properties,'dropOut')==0)
        net.layers{k}.properties.dropOut=1;
    end
    
    if (isfield(net.layers{k}.properties,'Activation')==0)
        net.layers{k}.properties.Activation=@Sigmoid;
    end
    if (isfield(net.layers{k}.properties,'dActivation')==0)
        net.layers{k}.properties.dActivation=@dSigmoid;
    end
    
    assert(((net.layers{k}.properties.dropOut<=1) &&(net.layers{k}.properties.dropOut>0)) ,'Dropout must be >0 and <=1 in layer %d',k);
    
    switch net.layers{k}.properties.type
        case net.types.input
            continue;
        case net.types.softmax
            net.layers{k}.properties.sizeFm = net.layers{k-1}.properties.sizeFm;
        case net.types.fc
            net.layers{k}.properties.sizeFm = 1;
        case net.types.batchNorm
            net.layers{k}.properties.sizeFm = net.layers{k-1}.properties.sizeFm;
        case net.types.reshape
            net.layers{k}.properties.sizeFm  = [net.layers{k}.properties.sizeFm 1 1 1];
            net.layers{k}.properties.sizeFm  = net.layers{k}.properties.sizeFm(1:3);
            assert(   prod(net.layers{k}.properties.sizeFm)*net.layers{k}.properties.numFm == prod(net.layers{k-1}.properties.sizeOut), 'Error - reshape must have the same num of elements as the layer before (%d != %d), layer %d\n',prod(net.layers{k}.properties.sizeFm)*net.layers{k}.properties.numFm , prod(net.layers{k-1}.properties.sizeOut),k);
        case net.types.conv
            net.layers{k}.properties.inputDim = max(1,sum(net.layers{k-1}.properties.sizeFm>1));
            assert( ((isfield(net.layers{k}.properties,'pad')==0)    || (length(net.layers{k}.properties.pad)==1)     || (length(net.layers{k}.properties.pad)==net.layers{k}.properties.inputDim) )    , 'Error - pad can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            assert( ((isfield(net.layers{k}.properties,'stride')==0) || (length(net.layers{k}.properties.stride)==1)  || (length(net.layers{k}.properties.stride)==net.layers{k}.properties.inputDim) ) , 'Error - stride can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            assert( ((isfield(net.layers{k}.properties,'pooling')==0)|| (length(net.layers{k}.properties.pooling)==1) || (length(net.layers{k}.properties.pooling)==net.layers{k}.properties.inputDim) ), 'Error - pooling can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            
            if (isfield(net.layers{k}.properties,'stride')==0)
                net.layers{k}.properties.stride=ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            if (isfield(net.layers{k}.properties,'pad')==0)
                net.layers{k}.properties.pad=zeros(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            if (isfield(net.layers{k}.properties,'pooling')==0)
                net.layers{k}.properties.pooling=ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            %sanity checks
            assert( isfield(net.layers{k}.properties,'kernel')==1, 'Error - missing kernel definition in layer %d',k);
            
            
            if (isscalar(net.layers{k}.properties.kernel))
                net.layers{k}.properties.kernel = net.layers{k}.properties.kernel*ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            
            %pad default settings
            if (isscalar(net.layers{k}.properties.pad))
                net.layers{k}.properties.pad = net.layers{k}.properties.pad*ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            net.layers{k}.properties.pad     = [net.layers{k}.properties.pad 0 0 0];
            net.layers{k}.properties.pad     = net.layers{k}.properties.pad(1:3);

            net.layers{k}.properties.kernel  = [net.layers{k}.properties.kernel 1 1 1];
            net.layers{k}.properties.kernel  = net.layers{k}.properties.kernel(1:3);
            
            
            %pooling default settings
            if (isscalar(net.layers{k}.properties.pooling))
                net.layers{k}.properties.pooling = net.layers{k}.properties.pooling*ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            net.layers{k}.properties.pooling = [net.layers{k}.properties.pooling 1 1 1];
            net.layers{k}.properties.pooling = net.layers{k}.properties.pooling(1:3);
            net.layers{k}.properties.pooling = min(net.layers{k}.properties.pooling ,net.layers{k-1}.properties.sizeFm);
            
            %stride default settings
            if (isscalar(net.layers{k}.properties.stride))
                net.layers{k}.properties.stride = net.layers{k}.properties.stride*ones(1,sum(net.layers{k-1}.properties.sizeFm>1));
            end
            net.layers{k}.properties.stride  = [net.layers{k}.properties.stride 1 1 1];
            net.layers{k}.properties.stride  = net.layers{k}.properties.stride(1:3);
            net.layers{k}.properties.stride = min(net.layers{k}.properties.stride ,net.layers{1}.properties.sizeFm);
            
            
            assert( isempty(find(net.layers{k}.properties.pooling<1, 1)), 'Error - pooling must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.kernel<1, 1)) , 'Error - kernel must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.stride<1, 1)) , 'Error - stride must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.pad<0, 1))    , 'Error - pad must be >=0 for all dimensions, layer=%d',k);
            assert( (net.layers{k}.properties.dropOut<=1 && net.layers{k}.properties.dropOut>0), 'Error - dropOut must be >0 and <=1, layer=%d, dropOut=%d',k,net.layers{k}.properties.dropOut);
            assert( isempty(find(net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad<net.layers{k}.properties.kernel, 1)) , 'Error - kernel too large (%s), must be smaller then prev layer FM size (%s) plus pad (%s), layer=%d',...
                num2str(net.layers{k}.properties.kernel) , num2str(net.layers{k-1}.properties.sizeFm) , num2str(net.layers{k}.properties.pad) ,k );
            assert( isempty(find(net.layers{k}.properties.pad>=net.layers{k}.properties.kernel, 1)) , 'Error - pad too large (%s), must be smaller then kernel size (%s), layer=%d',...
                num2str(net.layers{k}.properties.pad),num2str(net.layers{k}.properties.kernel),k);
            
            [f,~] = log2(net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad);
            if (~isempty(find(f~=0.5, 1)))
                warning(['Layer ' num2str(k) ' input plus pad is ' ...
                    num2str(net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad) ...
                    ' , not a power of 2. May reduce speed']);
            end
            net.layers{k}.properties.sizeFm = ceil((floor((net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1)./net.layers{k}.properties.pooling);
            
    end
    
    
    if (~isequal(net.layers{k}.properties.type,net.types.conv)) % not conv
        numNewronsInPrevLayer = net.layers{k-1}.properties.numFm*prod(net.layers{k-1}.properties.sizeFm);
        numInputs=numNewronsInPrevLayer+1;
        if (isequal(net.layers{k}.properties.type,net.types.fc))
            net.layers{k}.fcweight = normrnd(0,1/sqrt(numInputs*prevLayerActivation),numInputs,net.layers{k}.properties.numFm);% add one for bias
            net.layers{k}.momentum = net.layers{k}.fcweight * 0;
            if (~isnan(net.hyperParam.constInitWeight))
                net.layers{k}.fcweight = net.layers{k}.fcweight*0+net.hyperParam.constInitWeight;
            end
            net.layers{k}.properties.numWeights = numel(net.layers{k}.fcweight);
        elseif (isequal(net.layers{k}.properties.type,net.types.batchNorm)) %batchnorm
           
        else
            net.layers{k}.properties.numWeights = 0; % softmax
        end
    else   % is conv layer
        net.layers{k}.properties.numWeights = 0;
        for fm=1:net.layers{k}.properties.numFm
            for prevFm=1:net.layers{k-1}.properties.numFm
                numInputs=net.layers{k-1}.properties.numFm*prod(net.layers{k}.properties.kernel)+1;
                net.layers{k}.weight{fm}(:,:,:,prevFm) = normrnd(0,1/sqrt(numInputs*prevLayerActivation),net.layers{k}.properties.kernel);
                net.layers{k}.momentum{fm}(:,:,:,prevFm) = net.layers{k}.weight{fm}(:,:,:,prevFm) * 0;
                if (~isnan(net.hyperParam.constInitWeight))
                    net.layers{k}.weight{fm}(:,:,:,prevFm)   = net.hyperParam.constInitWeight+0*net.layers{k}.weight{fm}(:,:,:,prevFm);
                end
                
                net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn(flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad));
                net.layers{k}.properties.numWeights = net.layers{k}.properties.numWeights + numel(net.layers{k}.weight{fm}(:,:,:,prevFm));
            end
        end
        fftWeightFlipped = conj(fftn(net.layers{k}.weight{1}(:,:,:,1) , (net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad)));
        net.layers{k}.flipMat = repmat(fftWeightFlipped./net.layers{k}.weightFFT{1}(:,:,:,1) , [1 1 1 net.layers{k-1}.properties.numFm]);
        
        %bias val
        numInputs=net.layers{k-1}.properties.numFm*prod(net.layers{k}.properties.kernel)+1;
        net.layers{k}.bias = normrnd(0,1/sqrt(numInputs*prevLayerActivation),net.layers{k}.properties.numFm,1);% add one for bias
        if (~isnan(net.hyperParam.constInitWeight))
            net.layers{k}.bias = net.hyperParam.constInitWeight+0*net.layers{k}.bias;
        end
        net.layers{k}.momentumBias = net.layers{k}.bias * 0 ;
        net.layers{k}.properties.numWeights = net.layers{k}.properties.numWeights + numel(net.layers{k}.bias);
        
        
        %%%%%% stride looksups , the below is used to speed performance
        for dim=1:3
            net.layers{k}.properties.indexesStride{dim} = net.layers{k}.properties.kernel(dim):net.layers{k}.properties.stride(dim):(net.layers{k-1}.properties.sizeFm(dim)+2*net.layers{k}.properties.pad(dim));
        end
        
        %%%%%% pooling looksups , the below is nasty code:) but done only during initialization
        if ( ~isempty(find(net.layers{k}.properties.pooling>1, 1))) %pooling exist
            net.layers{k}.properties.indexes=[];
            net.layers{k}.properties.indexesIncludeOutBounds=[];
            net.layers{k}.properties.indexesReshape=[];
            
            elemSize = prod(net.layers{k}.properties.pooling);
            net.layers{k}.properties.offsets = ((1:(prod([net.layers{k}.properties.sizeFm net.layers{k}.properties.numFm]))) -1 )*elemSize;
            %init some indexes for optimized access during
            %feedForward/Backprop
            ranges=floor((net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1;
            for fm=1:net.layers{k}.properties.numFm
                for row=1:prod(net.layers{k}.properties.sizeFm)
                    [y,x,z] = ind2sub(net.layers{k}.properties.sizeFm, row);
                    for col=1:prod(net.layers{k}.properties.pooling)
                        [yy,xx,zz] = ind2sub(net.layers{k}.properties.pooling, col);
                        net.layers{k}.properties.indexesIncludeOutBounds(end+1) = (fm-1)      *prod(ranges(1:3)) +...
                            ((zz-1)+(z-1)*net.layers{k}.properties.pooling(3))*prod(ranges(1:2)) +...
                            ((xx-1)+(x-1)*net.layers{k}.properties.pooling(2))*prod(ranges(1:1)) +...
                            ((yy-1)+(y-1)*net.layers{k}.properties.pooling(1)) + ...
                            1;
                        if ( isempty(find( ...
                                ((([yy xx zz]-1)+([y x z]-1).*net.layers{k}.properties.pooling)+1) > ranges, 1 )))
                            net.layers{k}.properties.indexes(end+1) = net.layers{k}.properties.indexesIncludeOutBounds(end);
                            net.layers{k}.properties.indexesReshape(end+1) = (col-1) + (row+(fm-1)*prod(net.layers{k}.properties.sizeFm)-1)*prod(net.layers{k}.properties.pooling) + 1;
                        end
                    end
                end
            end
        end
    end
    
    assert(isfield(net.layers{k}.properties,'sizeFm') , 'Error - missing sizeFm field in layer %d\n',k);
    assert(isfield(net.layers{k}.properties,'numFm') , 'Error - missing numFm field in layer %d\n',k);
    net.properties.numWeights = net.properties.numWeights + net.layers{k}.properties.numWeights;
    prevLayerActivation = net.layers{k}.properties.dropOut;
    net.layers{k}.properties.sizeOut = [net.layers{k}.properties.sizeFm net.layers{k}.properties.numFm];
end

net.layers{end}.properties.numFm = net.layers{end-1}.properties.numFm;
net.layers{end}.properties.numWeights = 0;

assert(net.layers{end-1}.properties.dropOut==1,'Last layer must be with dropout=1');

end

