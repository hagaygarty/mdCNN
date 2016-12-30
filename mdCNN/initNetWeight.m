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
    assert( isfield(net.hyperParam,'sizeFmInput'), 'Error - sizeFmInput is not defined in hyperParam. Example: for 32x32 rgb please set to [32 32] , numFmInput=3');
    assert( length(net.hyperParam.sizeFmInput)<=3, 'Error - sizeFmInput cannot have more then 3 dimensions');
    assert( isequal(net.hyperParam.sizeFmInput(net.hyperParam.sizeFmInput>0),net.hyperParam.sizeFmInput), 'Error - sizeFmInput cannot have dim 0');
    assert( ((length(net.hyperParam.sizeFmInput)==1) || (isequal(net.hyperParam.sizeFmInput(net.hyperParam.sizeFmInput>1),net.hyperParam.sizeFmInput))), 'Error - sizeFmInput cannot have useless 1 dims');
    assert( isfield(net.hyperParam,'numFmInput'), 'Error - numFmInput is not defined in hyperParam. Example: for 32x32 rgb please set to numFmInput=3');
    assert( isempty(find(net.hyperParam.sizeFmInput<1, 1)), 'Error - sizeFmInput must be >=1 for all dimensions');
    assert( ((~isempty(net.hyperParam.sizeFmInput)) && (length(net.hyperParam.sizeFmInput)<=3)), 'Error - num of dimensions must be >0 and <=3');
           
    net.hyperParam.sizeFmInput = [net.hyperParam.sizeFmInput 1 1 1];
    net.hyperParam.sizeFmInput = net.hyperParam.sizeFmInput(1:3);

    net.properties.sizeInput  = net.hyperParam.sizeFmInput;
    net.properties.InputNumFm = net.hyperParam.numFmInput;

    if (isscalar(net.hyperParam.augmentParams.maxStride))
        net.hyperParam.augmentParams.maxStride = net.hyperParam.augmentParams.maxStride*ones(1,length(net.hyperParam.sizeFmInput)).*(net.hyperParam.sizeFmInput>1);
    end
            
    for k=1:size(net.layers,2)
        fprintf('Initializing layer %d\n',k);
        assert( isfield(net.layers{k}.properties,'type')==1, 'Error - missing type definition in layer %d',k);
        assert( isfield(net.layers{k}.properties,'numFm')==1, 'Error - missing numFM definition in layer %d',k);
        
        if (k==1)
            net.layers{k}.properties.numFmInPrevLayer  = net.hyperParam.numFmInput;            
            net.layers{k}.properties.sizeFmInPrevLayer = net.hyperParam.sizeFmInput;
        else
            net.layers{k}.properties.numFmInPrevLayer  = net.layers{k-1}.properties.numFm;
            net.layers{k}.properties.sizeFmInPrevLayer = net.layers{k-1}.properties.out;
        end

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
        
        if ( net.layers{k}.properties.type==1) % is fully connected layer
            net.layers{k}.properties.out = 1;
        else
            net.layers{k}.properties.inputDim = sum(net.layers{k}.properties.sizeFmInPrevLayer>1);
            assert( ((isfield(net.layers{k}.properties,'pad')==0)    || (length(net.layers{k}.properties.pad)==1)     || (length(net.layers{k}.properties.pad)==net.layers{k}.properties.inputDim) )    , 'Error - pad can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            assert( ((isfield(net.layers{k}.properties,'stride')==0) || (length(net.layers{k}.properties.stride)==1)  || (length(net.layers{k}.properties.stride)==net.layers{k}.properties.inputDim) ) , 'Error - stride can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            assert( ((isfield(net.layers{k}.properties,'pooling')==0)|| (length(net.layers{k}.properties.pooling)==1) || (length(net.layers{k}.properties.pooling)==net.layers{k}.properties.inputDim) ), 'Error - pooling can be a scalar or a vector with length as num dimnetions (%d), layer=%d',net.layers{k}.properties.inputDim,k);
            
            if (isfield(net.layers{k}.properties,'stride')==0)
                net.layers{k}.properties.stride=ones(1,length(net.hyperParam.sizeFmInput));
            end
            if (isfield(net.layers{k}.properties,'pad')==0)
                net.layers{k}.properties.pad=zeros(1,length(net.hyperParam.sizeFmInput));
            end
            if (isfield(net.layers{k}.properties,'pooling')==0)
                net.layers{k}.properties.pooling=ones(1,length(net.hyperParam.sizeFmInput));
            end
            %sanity checks
            assert( isfield(net.layers{k}.properties,'kernel')==1, 'Error - missing kernel definition in layer %d',k);

   
            if (isscalar(net.layers{k}.properties.kernel))
                net.layers{k}.properties.kernel = net.layers{k}.properties.kernel*ones(1,length(net.hyperParam.sizeFmInput));
            end
            net.layers{k}.properties.kernel = min(net.layers{k}.properties.kernel ,net.hyperParam.sizeFmInput);

            %pad default settings
            if (isscalar(net.layers{k}.properties.pad))
                net.layers{k}.properties.pad = net.layers{k}.properties.pad*ones(1,length(net.hyperParam.sizeFmInput));
            end
            net.layers{k}.properties.pad = net.layers{k}.properties.pad.*(net.hyperParam.sizeFmInput>1);
            
            %pooling default settings
            if (isscalar(net.layers{k}.properties.pooling))
                net.layers{k}.properties.pooling = net.layers{k}.properties.pooling*ones(1,length(net.hyperParam.sizeFmInput));
            end
            net.layers{k}.properties.pooling = min(net.layers{k}.properties.pooling ,net.hyperParam.sizeFmInput);

            %stride default settings
            if (isscalar(net.layers{k}.properties.stride))
                net.layers{k}.properties.stride = net.layers{k}.properties.stride*ones(1,length(net.hyperParam.sizeFmInput));
            end
            net.layers{k}.properties.stride = min(net.layers{k}.properties.stride ,net.hyperParam.sizeFmInput);
               
            net.layers{k}.properties.stride  = [net.layers{k}.properties.stride 1 1 1];
            net.layers{k}.properties.stride  = net.layers{k}.properties.stride(1:3);
            net.layers{k}.properties.pad     = [net.layers{k}.properties.pad 1 1 1];
            net.layers{k}.properties.pad     = net.layers{k}.properties.pad(1:3);
            net.layers{k}.properties.pooling = [net.layers{k}.properties.pooling 1 1 1];
            net.layers{k}.properties.pooling = net.layers{k}.properties.pooling(1:3);
            
            assert( isempty(find(net.layers{k}.properties.pooling<1, 1)), 'Error - pooling must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.kernel<1, 1)) , 'Error - kernel must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.stride<1, 1)) , 'Error - stride must be >=1 for all dimensions, layer=%d',k);
            assert( isempty(find(net.layers{k}.properties.pad<0, 1))    , 'Error - pad must be >=0 for all dimensions, layer=%d',k);
            assert( (net.layers{k}.properties.dropOut<=1 && net.layers{k}.properties.dropOut>0), 'Error - dropOut must be >0 and <=1, layer=%d, dropOut=%d',k,net.layers{k}.properties.dropOut);
            assert( isempty(find(net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad<net.layers{k}.properties.kernel, 1)) , 'Error - kernel too large (%s), must be smaller then prev layer FM size (%s) plus pad (%s), layer=%d',...
                num2str(net.layers{k}.properties.kernel) , num2str(net.layers{k}.properties.sizeFmInPrevLayer) , num2str(net.layers{k}.properties.pad) ,k );
            assert( isempty(find(net.layers{k}.properties.pad>=net.layers{k}.properties.kernel, 1)) , 'Error - pad too large (%s), must be smaller then kernel size (%s), layer=%d',...
                num2str(net.layers{k}.properties.pad),num2str(net.layers{k}.properties.kernel),k);

            [f,~] = log2(net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad);
            if (~isempty(find(f~=0.5, 1)))
                warning(['Layer ' num2str(k) ' input plus pad is ' ...
                    num2str(net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad) ...
                    ' , not a power of 2. Might reduce speed']);
            end
            net.layers{k}.properties.out = ceil((floor((net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1)./net.layers{k}.properties.pooling);
        end
        

        
        if (net.layers{k}.properties.type==1) % is fully connected layer
            numNewronsInPrevLayer = net.layers{k}.properties.numFmInPrevLayer*prod(net.layers{k}.properties.sizeFmInPrevLayer);
            numInputs=numNewronsInPrevLayer+1;
            net.layers{k}.fcweight = normrnd(0,1/sqrt(numInputs*prevLayerActivation),numInputs,net.layers{k}.properties.numFm);% add one for bias
            net.layers{k}.momentum = net.layers{k}.fcweight * 0;
            if (~isnan(net.hyperParam.constInitWeight))
                net.layers{k}.fcweight = net.layers{k}.fcweight*0+net.hyperParam.constInitWeight;
            end
            net.layers{k}.dW = zeros(size(net.layers{k}.fcweight));
            net.layers{k}.properties.numWeights = numel(net.layers{k}.fcweight);
        else   % is conv layer
            net.layers{k}.properties.numWeights = 0;
            for fm=1:net.layers{k}.properties.numFm
                for prevFm=1:net.layers{k}.properties.numFmInPrevLayer
                    numInputs=net.layers{k}.properties.numFmInPrevLayer*prod(net.layers{k}.properties.kernel)+1;
                    net.layers{k}.weight{fm}(:,:,:,prevFm) = normrnd(0,1/sqrt(numInputs*prevLayerActivation),net.layers{k}.properties.kernel);
                    net.layers{k}.momentum{fm}(:,:,:,prevFm) = net.layers{k}.weight{fm}(:,:,:,prevFm) * 0;
                    if (~isnan(net.hyperParam.constInitWeight))
                        net.layers{k}.weight{fm}(:,:,:,prevFm)   = net.hyperParam.constInitWeight+0*net.layers{k}.weight{fm}(:,:,:,prevFm);
                    end
                    
                    net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn(flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad));
                    net.layers{k}.dW{fm}(:,:,:,prevFm) = zeros(size(net.layers{k}.weight{fm}(:,:,:,prevFm)));
                    net.layers{k}.properties.numWeights = net.layers{k}.properties.numWeights + numel(net.layers{k}.weight{fm}(:,:,:,prevFm));
                end
            end
            fftWeightFlipped = conj(fftn(net.layers{k}.weight{1}(:,:,:,1) , (net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad)));
            net.layers{k}.flipMat = repmat(fftWeightFlipped./net.layers{k}.weightFFT{1}(:,:,:,1) , [1 1 1 net.layers{k}.properties.numFmInPrevLayer]);
            
            %bias val
            numInputs=net.layers{k}.properties.numFmInPrevLayer*prod(net.layers{k}.properties.kernel)+1;
            net.layers{k}.bias = normrnd(0,1/sqrt(numInputs*prevLayerActivation),1,net.layers{k}.properties.numFm);% add one for bias   
            net.layers{k}.biasdW = zeros(size(net.layers{k}.bias));
            if (~isnan(net.hyperParam.constInitWeight))
                net.layers{k}.bias = net.hyperParam.constInitWeight+0*net.layers{k}.bias;
            end
            net.layers{k}.momentumBias = net.layers{k}.bias * 0 ;
            net.layers{k}.properties.numWeights = net.layers{k}.properties.numWeights + numel(net.layers{k}.bias);

            
            %%%%%% stride looksups , the below is used to speed performance
            for dim=1:3
                net.layers{k}.properties.indexesStride{dim} = net.layers{k}.properties.kernel(dim):net.layers{k}.properties.stride(dim):(net.layers{k}.properties.sizeFmInPrevLayer(dim)+2*net.layers{k}.properties.pad(dim));
            end
            
            %%%%%% pooling looksups , the below is nasty code:) but done only during initialization
            if ( ~isempty(find(net.layers{k}.properties.pooling>1, 1))) %pooling exist
                net.layers{k}.properties.indexes=[];
                net.layers{k}.properties.indexesIncludeOutBounds=[];
                net.layers{k}.properties.indexesReshape=[];

                elemSize = prod(net.layers{k}.properties.pooling);
                net.layers{k}.properties.offsets = ((1:(prod([net.layers{k}.properties.out net.layers{k}.properties.numFm]))) -1 )*elemSize;
                %init some indexes for optimized access during
                %feedForward/Backprop
                ranges=floor((net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad-net.layers{k}.properties.kernel)./net.layers{k}.properties.stride)+1;
                for fm=1:net.layers{k}.properties.numFm
                    for row=1:prod(net.layers{k}.properties.out)
                        [y,x,z] = ind2sub(net.layers{k}.properties.out, row);
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
                                net.layers{k}.properties.indexesReshape(end+1) = (col-1) + (row+(fm-1)*prod(net.layers{k}.properties.out)-1)*prod(net.layers{k}.properties.pooling) + 1;
                            end
                        end
                    end
                end
             end
        end

        net.properties.numWeights = net.properties.numWeights + net.layers{k}.properties.numWeights;
        prevLayerActivation = net.layers{k}.properties.dropOut;
    end
	
	assert(net.layers{end}.properties.type==1,'Last layer must be FC layer');
	assert(net.layers{end}.properties.dropOut==1,'Last layer must be with dropout=1');

end

