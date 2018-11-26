%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ ] = verifyBackProp(net)
%% verification of network correctnes. Search for implementation errors by verifying the derivitives calculated by backProp are correct
%% the function will verify backProp on some of the weights, (selected randomly)

initSeed = rng;

fprintf('Verifying backProp..');


batchNum=net.hyperParam.batchNum;


input = normrnd(0,1, [net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm batchNum]);

expectedOut=rand(net.layers{end}.properties.numFm,batchNum);

seed = rng;
% create calculated dCdW
net = backPropegate(net, input, expectedOut);

%create W+dW
dw=1e-7;
th = 1e-5;
numIter=1;

startVerification=clock;

for k=2:size(net.layers,2)-1
    if (isequal(net.layers{k}.properties.type,net.types.softmax))
        continue;
    end
    if (isequal(net.layers{k}.properties.type,net.types.batchNorm))
        continue;
    end
    
    estimateddActivation = (net.layers{k}.properties.Activation(1+dw)-net.layers{k}.properties.Activation(1))/dw;
    realdActivation = net.layers{k}.properties.dActivation(1);
    diff = estimateddActivation-realdActivation;
    if ( abs(diff) > th)
        assert(0,'Activation and dActivation do not match!');
    end
    
    estimatedLoss = ( net.layers{end}.properties.costFunc(1+dw,1)-net.layers{end}.properties.costFunc(1,1) )/dw;
    realdLoss = net.layers{end}.properties.lossFunc(1,1);
    diff = estimatedLoss-realdLoss;
    if ( abs(diff) > th)
        assert(0,'costFunc and lossFunc do not match! Should be: lossFunc = d/dx costFunc');
    end
    
    
    for fm=1:net.layers{k}.properties.numFm
        if (isequal(net.layers{k}.properties.type,net.types.fc))
            numprevFm = 1;
        else
            numprevFm = size(net.layers{k}.weight{1},4);
        end
        for prevFm=1:numprevFm
            for iter=1:numIter
                if (isequal(net.layers{k}.properties.type,net.types.fc))
                    y=randi(size(net.layers{k}.fcweight,1));
                    x=randi(size(net.layers{k}.fcweight,2));
                    calculatedDcDw = net.layers{k}.dW(y,x);
                else
                    y=randi(size(net.layers{k}.weight{1},1));
                    x=randi(size(net.layers{k}.weight{1},2));
                    z=randi(size(net.layers{k}.weight{1},3));
                    calculatedDcDw = net.layers{k}.dW{fm}(y,x,z,prevFm);
                end
                
                rng(seed);%to set the same dropout each time..
                net = feedForward(net, input, 0);
                if (isequal(net.layers{k}.properties.type,net.types.fc))
                    net.layers{k}.fcweight(y,x) = net.layers{k}.fcweight(y,x) + dw;
                else
                    net.layers{k}.weight{fm}(y,x,z,prevFm) = net.layers{k}.weight{fm}(y,x,z,prevFm) + dw;
                    net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad));
                end
                
                rng(seed);%to set the same dropout each time..
                netPdW =  feedForward(net, input, 0);
                if (isequal(net.layers{k}.properties.type,net.types.fc))
                    net.layers{k}.fcweight(y,x) = net.layers{k}.fcweight(y,x) - dw;
                else
                    net.layers{k}.weight{fm}(y,x,z,prevFm) = net.layers{k}.weight{fm}(y,x,z,prevFm) - dw;
                    net.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(net.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (net.layers{k-1}.properties.sizeFm+2*net.layers{k}.properties.pad));
                end
                
                cW = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
                cWPlusDw = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
                
                estimatedDcDw = (cWPlusDw-cW)/dw;
                diff = estimatedDcDw-calculatedDcDw;
                if ( abs(sum(diff)) > th )
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
for k=2:size(net.layers,2)-1
    if (~isequal(net.layers{k}.properties.type,net.types.conv)) % not for conv
        continue;
    end
    
    for fm=1:net.layers{k}.properties.numFm
        
        calculatedDcDw = net.layers{k}.biasdW(fm);
        rng(seed);%to set the same dropout each time..
        net       =  feedForward(net, input, 0);
        net.layers{k}.bias(fm) = net.layers{k}.bias(fm) + dw;
        rng(seed);%to set the same dropout each time..
        netPdW =  feedForward(net, input, 0);
        net.layers{k}.bias(fm) = net.layers{k}.bias(fm) - dw;
        
        cW = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);
        cWPlusDw = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
        
        estimatedDcDw = (cWPlusDw-cW)/dw;
        
        diff = estimatedDcDw-calculatedDcDw;
        if ( abs(sum(diff)) > th)
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

