function [  ] = displayFilters(net , image , label)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (exist('label','var')==0)
    label='Unknown';
end


figure('Name' , ['Network filters, ' label]);

input = GetNetworkInputs(image, net, 1);
outs = feedForward(net.layers, input , 1);

imagePerXAxe=12;
imagePerYAxe=size(outs,1)+1;

subplot(imagePerYAxe,imagePerXAxe,1,'replace');
axis off
imshow(image);
title(['Input: ' label]);
subplot(imagePerYAxe,imagePerXAxe,2,'replace');
axis off
input = squeeze(input);
imshow(input);
title('Normalized Input');

for k=1:size(net.layers,2)
    layerMin = Inf; layerMax = -Inf;
    for fm=1:min(imagePerXAxe,net.layers{k}.properties.numFm)
        if (net.layers{k}.properties.type==1) % is fully connected layer
            im = outs{k}.z(:,fm,1);        
        else
            im = outs{k}.z(:,:,fm);        
        end
         layerMin = min ( min(min(im)) , layerMin);
         layerMax = max ( max(max(im)) , layerMax);
    end
    
    for fm=1:min(imagePerXAxe,net.layers{k}.properties.numFm)
         if ( net.layers{k}.properties.type==1) % is fully connected layer
            im = outs{k}.z(:,fm,1);        
         else
            im = outs{k}.z(:,:,fm);        
         end       
         im=im-layerMin;
         im = im/(layerMax-layerMin);
         subplot(imagePerYAxe,imagePerXAxe,k*imagePerXAxe+fm,'replace');
         axis off
         imshow(im);
         if (fm==1)
             if (k==size(outs,1))
                title(['Out - Layer ' num2str(k)]);
             else
                title(['Layer ' num2str(k)]);
             end
         end
    end
end

drawnow;

end

