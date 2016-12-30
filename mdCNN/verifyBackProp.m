%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ ] = verifyBackProp(net)
%% verification of network correctnes. Search for implementation errors by verifying the derivitives calculated by backProp are correct
%% the function will verify backProp on some of the weights, (selected randomly)

initSeed = rng;

fprintf('Verifying backProp..');

errorMethod = 1;

if (net.layers{1}.properties.type==1) % is fully connected layer 
    input = normrnd(0,1, 1 , size(net.layers{1}.fcweight,1)-1);
else
    input = normrnd(0,1,[net.layers{1}.properties.sizeFmInPrevLayer net.layers{1}.properties.numFmInPrevLayer]);
end
expectedOut=rand(1,net.layers{end}.properties.numFm);

seed = rng;
% create calculated dCdW
net = backPropegate(net, input, expectedOut);
net_WdCdW = updateWeights(net, 1 , 0  , 0);

%create W+dW
dw=0.0000001;
th = 0.0001;
numIter=1;

startVerification=clock;

for k=1:size(net.layers,2)
    estimateddActivation = (net.layers{k}.properties.Activation(1+dw)-net.layers{k}.properties.Activation(1))/dw;
    realdActivation = net.layers{k}.properties.dActivation(1);
    diff = estimateddActivation-realdActivation;
    if ( abs(diff) > th)
        assert(0,'Activation and dActivation do not match!');
    end
    
    for fm=1:net.layers{k}.properties.numFm
        if (net.layers{k}.properties.type==1)
            numprevFm = 1;
        else
            numprevFm = size(net.layers{k}.weight{1},4);
        end
        for prevFm=1:numprevFm
            for iter=1:numIter
                if (net.layers{k}.properties.type==1)
                    y=randi(size(net.layers{k}.fcweight,1));
                    x=randi(size(net.layers{k}.fcweight,2));
                    calculatedDcDw = net.layers{k}.fcweight(y,x) - net_WdCdW.layers{k}.fcweight(y,x);
                else
                    y=randi(size(net.layers{k}.weight{1},1));
                    x=randi(size(net.layers{k}.weight{1},2));
                    z=randi(size(net.layers{k}.weight{1},3));
                    calculatedDcDw = net.layers{k}.weight{fm}(y,x,z,prevFm) - net_WdCdW.layers{k}.weight{fm}(y,x,z,prevFm);
                end

                rng(seed);%to set the same dropout each time..
                aW = feedForward(net.layers, input, 0);
                if (net.layers{k}.properties.type==1)
                    net.layers{k}.fcweight(y,x) = net.layers{k}.fcweight(y,x) + dw;
                else
                    net.layers{k}.weight{fm}(y,x,z,prevFm) = net.layers{k}.weight{fm}(y,x,z,prevFm) + dw;
                    net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad));
                end
                                
                rng(seed);%to set the same dropout each time..
                aWPlusDw =  feedForward(net.layers, input, 0);
                if (net.layers{k}.properties.type==1)
                    net.layers{k}.fcweight(y,x) = net.layers{k}.fcweight(y,x) - dw;
                else
                    net.layers{k}.weight{fm}(y,x,z,prevFm) = net.layers{k}.weight{fm}(y,x,z,prevFm) - dw;
                    net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k}.properties.sizeFmInPrevLayer+2*net.layers{k}.properties.pad));
                end
                   
                if (errorMethod==1)
                    cW = -sum((expectedOut).*log(aW{end}.activation) + (1-expectedOut).*log(1-aW{end}.activation));
                    cWPlusDw = -sum((expectedOut).*log(aWPlusDw{end}.activation) + (1-expectedOut).*log(1-aWPlusDw{end}.activation));
                else
                    cW       = 0.5*sum((aW{end}.activation-expectedOut).^2);
                    cWPlusDw = 0.5*sum((aWPlusDw{end}.activation-expectedOut).^2);
                end
                

                estimatedDcDw = (cWPlusDw-cW)/dw;
                diff = estimatedDcDw-calculatedDcDw;
                if ( abs(diff) > th)
                    if ( k == size(net.layers,2))
                        assert(0,'Big big blunder!!');
                    end
                    assert(0,'How come?? , problem in layer %d',k);
                end
            end
        end
    end
end

%check bias weight
for k=1:size(net.layers,2)
    if (net.layers{k}.properties.type==1) % is fully connected layer
        continue; %tested already in upper section
    end
    
    for fm=1:net.layers{k}.properties.numFm

        calculatedDcDw = net.layers{k}.bias(fm) - net_WdCdW.layers{k}.bias(fm);
        rng(seed);%to set the same dropout each time..
        aW       =  feedForward(net.layers, input, 0);
        net.layers{k}.bias(fm) = net.layers{k}.bias(fm) + dw;
        rng(seed);%to set the same dropout each time..
        aWPlusDw =  feedForward(net.layers, input, 0);
        net.layers{k}.bias(fm) = net.layers{k}.bias(fm) - dw;

        if (errorMethod==1)
            cW = -sum((expectedOut).*log(aW{end}.activation) + (1-expectedOut).*log(1-aW{end}.activation)); 
            cWPlusDw = -sum((expectedOut).*log(aWPlusDw{end}.activation) + (1-expectedOut).*log(1-aWPlusDw{end}.activation)); 
        else
            cW       = 0.5*sum((aW{end}.activation-expectedOut).^2);
            cWPlusDw = 0.5*sum((aWPlusDw{end}.activation-expectedOut).^2);
        end

        estimatedDcDw = (cWPlusDw-cW)/dw;

        diff = estimatedDcDw-calculatedDcDw;
        if ( abs(diff) > th)
            if ( k == size(net.layers,2) )
                assert(0,'Big big blunder!!');
            end
            assert(0,'How come??');
        end
    end
end

endVerification=clock;

fprintf('Network is OK. Verification time=%.2f\n',etime(endVerification,startVerification) );

rng(initSeed);%revert seed

end

