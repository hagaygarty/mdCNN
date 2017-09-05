%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ image ] = GetNetworkInputs(image , net , testTime)
%% function manipulate a sample and preperes it for passing into the net first layer
%% preperation can be bias removal , chane to variance 1 , scaling or patch selection , depending on the configuration


inputDim = net.hyperParam.sizeFmInput;
inputDim(end+1) = net.hyperParam.numFmInput;

if ( (inputDim(end-1)==1) && (ndims(image)==3) && (net.hyperParam.numFmInput~=size(image,ndims(image)))) 
    image = rgb2gray(image); %TODO - for color images? color 3d Images??
end

image=double(image);
if (net.hyperParam.numFmInput~=1)
    sz=size(image);
    singleFmDim=[sz(1:end-1) 1 1 1];
    singleFmDim = singleFmDim(1:3);
    image = reshape(image , [singleFmDim net.hyperParam.numFmInput] );
end

if (net.hyperParam.normalizeNetworkInput==1)
	for fm=1:net.hyperParam.numFmInput
		singleFm=image(:,:,:,fm);
		singleFm = singleFm-min(singleFm(:));
		
		if ( isfield(net.hyperParam.augmentParams, 'medianFilt') && (net.hyperParam.augmentParams.medianFilt > 0) )
			data = sort(singleFm(:));
			th = data(floor((length(data)-1)*net.hyperParam.augmentParams.medianFilt+1));
			singleFm(singleFm<th) = 0;
		end
	   
		maxImg=max(singleFm(:));
		if ( maxImg~=0 )
			singleFm = singleFm/maxImg;
		end
		image(:,:,:,fm) = singleFm;
	end
end

if (net.hyperParam.centralizeImage) && (~testTime)
    cm = round(centerOfMass(image))+1;
    tmpImage = padarray(image, ceil(size(image)/2),image(1,1),'replicate');
    image = tmpImage( cm(1):(cm(1)-1+size(image,1)) , cm(2):(cm(2)-1+size(image,2)));
end

if (net.hyperParam.cropImage) && (~testTime)
    maxBrightness=1/10;
    thFordetection = 0.03;
    outOfBounds = image >= maxBrightness;
    goodLines=find(sum(outOfBounds,2) > thFordetection * size(image,2));
    if (isempty(goodLines))
        goodLines = 1:size(image,1);
    end
    goodRows=find(sum(outOfBounds,1) > thFordetection * size(image,1));
    if (isempty(goodRows))
        goodRows = 1:size(image,2);
    end
    image([1:goodLines(1)-1 goodLines(end)+1:end],:) = [];
    image(:,[1:goodRows(1)-1 goodRows(end)+1:end])   = [];
    % border=2;
    % image = padarray(image, [border border],image(1,1),'replicate');
end

if (net.hyperParam.useRandomPatch>0)
    varImage=-Inf;
    iter=0; maxIter=1000;
    origIm=image;
    
    while(varImage<net.hyperParam.selevtivePatchVarTh)
        image = origIm;
       % patchSize=net.hyperParam.sizeFmInput;
       % patchSize = [patchSize(patchSize>1) net.hyperParam.numFmInput];

        patchSize = inputDim(1:3);
        
        szFm = [size(image) 1 1 1];
        szFm = szFm(1:3);

        image = padarray(image,max(0,patchSize-szFm));

        szFm = [size(image) 1 1 1];
        szFm = szFm(1:3);

        maxStride = max(1,szFm-patchSize+1);
        if ((testTime) && (net.hyperParam.testOnMiddlePatchOnly==1))
            starts = round(maxStride/2);%during test take only the middle patch
        else
            starts = arrayfun(@randi,maxStride);
        end
        ends = starts+patchSize-1;
        image = image(starts(1):ends(1) , starts(2):ends(2) , starts(3):ends(3),:);

        firstFm = image(:,:,:,1);
        varImage = var(firstFm(:));
        iter=iter+1;
        if ( iter>1 )
           % fprintf('\nSearch iter %d, size=%s, th=%f\n',iter, num2str(size(origIm)),net.hyperParam.selevtivePatchVarTh );
        end

        if (iter>=maxIter)
            fprintf('\ncouldn''t find patch in an image after %d tries, size=%s, th=%f\n',maxIter, num2str(size(origIm)),net.hyperParam.selevtivePatchVarTh);
            %assert(iter<maxIter , 'How come? bad image?\n');
            break;       
        end
    end
end

image = imresize3d(image, inputDim);


if (net.hyperParam.normalizeNetworkInput==1)
    varFact = sqrt(var(image(:)));
    if (varFact==0)
        varFact=1;
    end
    image = (image-mean(image(:))) /varFact;%normlize to mean 0 and var=1
end

if (net.hyperParam.flipImage==1) && (~testTime)
    for dim=length(find(net.hyperParam.sizeFmInput>1)):-1:1
        if (randi(2)==1)
            image = flip(image,dim);
        end
    end
end

if (net.hyperParam.numFmInput~=1)
    image = reshape(image , inputDim);
end

end

