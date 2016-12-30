function [ image ] = imresize3d( image , desiredDim)
if ( size(image) == desiredDim(1:ndims(image)) )
    return
end

if (ndims(image)<3)
    image = imresize(image , desiredDim(1:ndims(image)) ,'bilinear','Antialiasing',true );
    return;
end

for z=1:size(image,3)
    reducedImagexy(:,:,z) = imresize(image(:,:,z) , desiredDim(1:2) ,'bilinear','Antialiasing',true );
end

for y=1:size(reducedImagexy,1)
    reducedImage(y,:,:) = imresize(squeeze(reducedImagexy(y,:,:)) , [size(reducedImagexy,2) desiredDim(3)] ,'bilinear','Antialiasing',true );
end

image = reducedImage;
end

