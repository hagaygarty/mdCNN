function [  ] = displayFilters(net)

figure('Name' , 'Filters');

imagePerXAxe=12;
imagePerYAxe=12;

for k=1:size(net.layers,2)
    if (net.layers{k}.properties.type==1) % is fully connected layer
        continue
    end
    figure('Name' , ['layer ' num2str(k)]);
    
    for fm=1:min(imagePerYAxe*imagePerXAxe,net.layers{k}.properties.numFm)
         subplot(imagePerYAxe,imagePerXAxe,fm,'replace');
         axis off
         imshow(squeeze(net.layers{k}.weight{fm}));
    end
end

drawnow;

end

