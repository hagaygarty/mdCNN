function [ image ] = imrotate3d( image, angle1 , angle2)
 
if ( angle1~=0)
    image = imrotate(image,angle1,'bilinear');
end

if ( angle2~=0)
    image = permute (imrotate(permute(image,[3,1,2]),angle2,'bilinear') , [2 3 1]);
end

end

