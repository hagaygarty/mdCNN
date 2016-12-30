function [ image, complement ] = manipulateImage( image ,noiseVar , maxAngle , maxScaleFactor, minScaleFactor , maxStride , maxSigma , imageComplement )

origSize=size(image);

if ( ndims(image)<3)
    origSize(end+1) = 1;
end



scaleFactor = rand(1,length(origSize))*(maxScaleFactor-minScaleFactor) + minScaleFactor;

if ( randi(5)~=1 ) ;
    image=imresize3d(image,ceil(scaleFactor.*origSize));
end

if ( randi(5)~=1 ) 
    angle1 = rand(1)*(2*maxAngle)-maxAngle;
    if (ndims(image)<3)
        angle2=0;
    else
        angle2 = rand(1)*(2*maxAngle)-maxAngle;
    end
    image = imrotate3d(image,angle1,angle2);
end

image = padarray(image, maxStride(1:ndims(image))+max(0,origSize(1:ndims(image))-size(image)),'replicate');

starty = round((size(image,1)-origSize(1))/2)+1+randi([-maxStride(1) maxStride(1)]);
startx = round((size(image,2)-origSize(2))/2)+1+randi([-maxStride(2) maxStride(2)]);
startz = round((size(image,3)-origSize(3))/2)+1+randi([-maxStride(3) maxStride(3)]);

image = image(starty:(starty+origSize(1)-1) ,startx:(startx+origSize(2)-1),startz:(startz+origSize(3)-1));


if ((noiseVar>0) && ( randi(10)~=1 ))
    image=imnoise(image,'gaussian',rand(1)/1.3,noiseVar);
end


if (( maxSigma>0) && (randi(5)~=1))
    sigma = maxSigma*(1-log(randi(128-1))/log(128));
    image = imfilter(image,fspecial('gaussian',[9 9],sigma));
end

if ((imageComplement==1)&&(randi(2)~=1))
    complement=1;
    image = imcomplement(image);
else
    complement=0;
end

end

