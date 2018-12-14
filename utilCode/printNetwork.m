%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ] = printNetwork( net )
 disp(struct2table(net.hyperParam));
 disp(struct2table(net.runInfoParam));
 
 for k=1:size(net.layers,2)
     fprintf('Layer %d: ',k);
     if (isfield(net.layers{k}.properties,'Activation'))
        fprintf('Activation=%s, dActivation=%s\n', func2str(net.layers{k}.properties.Activation) , func2str(net.layers{k}.properties.dActivation));
     elseif (isfield(net.layers{k}.properties,'lossFunc'))
        fprintf('lossFunc=%s, costFunc=%s\n', func2str(net.layers{k}.properties.lossFunc) , func2str(net.layers{k}.properties.costFunc));
     else
        fprintf('\n');
     end
     disp(struct2table(net.layers{k}.properties));
 end
 
 fprintf('Network properties:\n\n');
 disp(struct2table(net.properties));
end
