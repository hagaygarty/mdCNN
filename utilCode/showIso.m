function [  ] = showIso( image , isoLevel, drawMontage, figureHandle)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  if (~exist('drawMontage','var'))
     drawMontage=1;
  end
  
  if (~exist('figureHandle','var'))
     figureHandle=figure;
  end
  
  if (~exist('isoLevel','var'))
     isoLevel=0;
  end

  if (isoLevel==0)
     isoLevel=0.8;
  end

  
  if (drawMontage==1)
    subplot(1,2,1);
  end
  image = double(image);
  image = image-min(image(:));
  image = image/max(image(:));
  filtImage  = imgaussfilt3(image,1);
  ball = ones(1,1,1);
  filtImage = imdilate(filtImage,ball,'same');
  filtImage = imerode(filtImage,ball,'same');

  s=sort(filtImage(:));
  isoVal = s(round(length(s)*isoLevel));
  
colormap('default');
%Ds = smooth3(V2);
hiso = patch(isosurface(filtImage,isoVal),  'FaceColor',[1,.75,.65],   'EdgeColor','none');
light('Position',[-1 0 0],'Style','local')
isonormals(filtImage,hiso);
daspect([1,1,1])
view(-10,-75); axis tight
camlight 
lighting gouraud
title('3d view');

if (drawMontage==1)
    subplot(1,2,2);
    im=reshape(image,size(image,1),size(image,2),1,size(image,3));
    montage(im);
    title('2d slices');
end

drawnow
end

