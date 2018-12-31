%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [  ] = verifyBackProp(net)
%% verification of network correctnes. Search for implementation errors by verifying the derivitives calculated by backProp are correct
%% the function will verify backProp on some of the weights, (selected randomly)

initSeed = rng;

fprintf('Verifying backProp..\n');


batchNum=net.hyperParam.batchNum;


input = normrnd(0,1, [net.layers{1}.properties.sizeFm net.layers{1}.properties.numFm batchNum]);
net =  feedForward(net, input, 0);


expectedOut=net.layers{end}.outs.activation;
expectedOut(expectedOut>0.5) = expectedOut(expectedOut>0.5)*0.99;
expectedOut(expectedOut<0.5) = expectedOut(expectedOut<0.5)*1.01;
expectedOut(expectedOut==0) = 0.001;

rng(initSeed);
% create calculated dCdW
net = backPropagate(net, input, expectedOut);

%create W+dW
dw=1*10^-9.5;
th = 1e-4;
numIter=1;
diffs=cell(length(net.layers),1); grads=diffs; diffsBias=grads;
startVerification=clock;

Cost = net.layers{end}.properties.costFunc(net.layers{end}.outs.activation,expectedOut);


for k=1:size(net.layers,2)
    fprintf('Checking layer %-2d - %-10s   ',k,net.layers{k}.properties.type);
    estimateddActivation = (net.layers{k}.properties.Activation([-1 1]+dw)-net.layers{k}.properties.Activation([-1 1]));
    realdActivation = net.layers{k}.properties.dActivation([-1 1]);
    diffs{k}(end+1) = max(abs(estimateddActivation-realdActivation*dw))/dw;
    if ( diffs{k}(end) > th)
        assert(0,'Activation and dActivation do not match!');
    end
    
    estimatedLoss = ( net.layers{end}.properties.costFunc(1+dw,1)-net.layers{end}.properties.costFunc(1,1) );
    realdLoss = net.layers{end}.properties.lossFunc(1,1);
    diffs{k}(end+1) = abs(estimatedLoss-realdLoss*dw)/dw;
    if ( diffs{k}(end) > th)
        assert(0,'costFunc and lossFunc do not match! Should be: lossFunc = d/dx costFunc');
    end
    
    if (isequal(net.layers{k}.properties.type,net.types.batchNorm)) % only for batchnorm
        %check beta/gamma
        for fm=1:net.layers{k}.properties.numFm
            for iter=1:numIter
                curIdx = numel(net.layers{k}.dbeta)/net.layers{k}.properties.numFm * (fm-1) +  randi(numel(net.layers{k}.dbeta)/net.layers{k}.properties.numFm);
                calculatedDcDbeta = net.layers{k}.dbeta(curIdx);
                calculatedDcDGamma = net.layers{k}.dgamma(curIdx);
                
                grads{k}(end+1) = calculatedDcDbeta;
                grads{k}(end+1) = calculatedDcDGamma;
                
                % check beta
                netVerify       =  net;
                netVerify.layers{k}.beta(curIdx) = netVerify.layers{k}.beta(curIdx) + dw;
                seedBefore=rng; rng(initSeed);%to set the same dropout each time..
                netPdW =  feedForward(netVerify, input, 0);
                rng(seedBefore);
                
                CostPlusDbeta = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
                
                estimatedDcDbeta = CostPlusDbeta-Cost;
                diffs{k}(end+1) = abs(sum(estimatedDcDbeta(:))-calculatedDcDbeta*dw)/dw;
                if ( diffs{k}(end) > sqrt(numel(estimatedDcDbeta))*th)
                    assert(0,'problem in beta gradient');
                end
                
                % check gamma
                netVerify       =  net;
                netVerify.layers{k}.gamma(curIdx) = netVerify.layers{k}.gamma(curIdx) + dw;
                seedBefore=rng; rng(initSeed);%to set the same dropout each time..
                netPdW =  feedForward(netVerify, input, 0);
                rng(seedBefore);
                
                CostPlusDGamma = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
                
                estimatedDcDgamma = (CostPlusDGamma-Cost);
                
                diffs{k}(end+1) = abs(sum(estimatedDcDgamma(:))-calculatedDcDGamma*dw)/dw;
                if ( diffs{k}(end) > sqrt(numel(estimatedDcDgamma))*th)
                    assert(0,'problem in gamma gradient');
                end
            end
       end
        
    end
    
    
    
    if (net.layers{k}.properties.numWeights>0)
        %check weights - fc and conv
        
        
        if (~isequal(net.layers{k}.properties.type,net.types.batchNorm))
            % check first fm, last fm and up to 50 in between
            for fm=[1:ceil(net.layers{k}.properties.numFm/50):net.layers{k}.properties.numFm net.layers{k}.properties.numFm]
                if (isequal(net.layers{k}.properties.type,net.types.fc))
                    numprevFm = 1;
                else
                    numprevFm = size(net.layers{k}.weight{1},4);
                end
                for prevFm=[1:ceil(numprevFm/50):numprevFm numprevFm]
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
                        grads{k}(end+1) = calculatedDcDw;
                        
                        netPdW = net;
                        if (isequal(netPdW.layers{k}.properties.type,netPdW.types.fc))
                            netPdW.layers{k}.fcweight(y,x) = netPdW.layers{k}.fcweight(y,x) + dw;
                        else
                            netPdW.layers{k}.weight{fm}(y,x,z,prevFm) = netPdW.layers{k}.weight{fm}(y,x,z,prevFm) + dw;
                            netPdW.layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(netPdW.layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (netPdW.layers{k-1}.properties.sizeFm+2*netPdW.layers{k}.properties.pad));
                        end
                        
                        seedBefore = rng; rng(initSeed);%to set the same dropout each time..
                        netPdW =  feedForward(netPdW, input, 0);
                        rng(seedBefore);
                        
                        cWPlusDw = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
                        
                        estimatedDcDw = (cWPlusDw-Cost);
                        diffs{k}(end+1) = abs(sum(estimatedDcDw(:))-calculatedDcDw*dw)/dw;
                        if ( diffs{k}(end) > sqrt(numel(estimatedDcDw))*th )
                            assert(0,'Problem in weight. layer %d',k);
                        end
                    end
                end
            end
        end
    end
    
    
    
    if (isequal(net.layers{k}.properties.type,net.types.conv)) % only for conv
        %check bias weight
        for fm=1:net.layers{k}.properties.numFm
            
            calculatedDcDw = net.layers{k}.biasdW(fm);
            grads{k}(end+1) = calculatedDcDw;
            netPdW       =  net;
            netPdW.layers{k}.bias(fm) = netPdW.layers{k}.bias(fm) + dw;
            seedBefore = rng; rng(initSeed);%to set the same dropout each time..
            netPdW =  feedForward(netPdW, input, 0);
            rng(seedBefore);
            cWPlusDw = netPdW.layers{end}.properties.costFunc(netPdW.layers{end}.outs.activation,expectedOut);
            
            estimatedDcDw = (cWPlusDw-Cost);
            
            diffsBias{k}(end+1) = abs(sum(estimatedDcDw(:))-calculatedDcDw*dw)/dw;
            if ( diffsBias{k}(end) > sqrt(numel(estimatedDcDw))*th)
                assert(0,'Problem in bias weight. layer %d',k);
            end
        end
    end
    
    %fprintf('mean diff=%.2e,max diff=%.2e, var diff=%.2e, rmsGrad=%.2e,varGrad=%.2e\n',mean(diffs{k}),max(diffs{k}),var(diffs{k}),rms(grads{k}),var(grads{k}));
    fprintf('\n');
    
end

endVerification=clock;
fprintf('Network is OK. Verification time=%.2f\n',etime(endVerification,startVerification));

rng(initSeed);%revert seed

