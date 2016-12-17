%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2015-16 Hagay Garty.
% hagaygarty@gmail.com , mdCNN library
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ ] = verifyBackProp(layers)
%% verification of network correctnes. Search for implementation errors by verifying the derivitives calculated by backProp are correct
%% the function will verify backProp on some of the weights, (selected randomly)

initSeed = rng;

fprintf('Verifying backProp..');

errorMethod = 1;

if (layers{1}.properties.type==1) % is fully connected layer 
    input = normrnd(0,1, 1 , size(layers{1}.fcweight,1)-1);
else
    input = normrnd(0,1,[layers{1}.properties.sizeFmInPrevLayer layers{1}.properties.numFmInPrevLayer]);
end
expectedOut=rand(1,layers{end}.properties.numFm);

seed = rng;
% create calculated dCdW
WdCdW = backPropegate(layers, input, expectedOut, 1 , 0 , errorMethod , 0);

%create W+dW
dw=0.0000001;
th = 0.0001;
numIter=1;

startVerification=clock;

for k=1:size(layers,2)
    estimateddActivation = (layers{k}.properties.Activation(1+dw)-layers{k}.properties.Activation(1))/dw;
    realdActivation = layers{k}.properties.dActivation(1);
    diff = estimateddActivation-realdActivation;
    if ( abs(diff) > th)
        assert(0,'Activation and dActivation are not the same!');
    end
    
    for fm=1:layers{k}.properties.numFm
        if (layers{k}.properties.type==1)
            numprevFm = 1;
        else
            numprevFm = size(layers{k}.weight{1},4);
        end
        for prevFm=1:numprevFm
            for iter=1:numIter
                if (layers{k}.properties.type==1)
                    y=randi(size(layers{k}.fcweight,1));
                    x=randi(size(layers{k}.fcweight,2));
                    calculatedDcDw = layers{k}.fcweight(y,x) - WdCdW{k}.fcweight(y,x);
                else
                    y=randi(size(layers{k}.weight{1},1));
                    x=randi(size(layers{k}.weight{1},2));
                    z=randi(size(layers{k}.weight{1},3));
                    calculatedDcDw = layers{k}.weight{fm}(y,x,z,prevFm) - WdCdW{k}.weight{fm}(y,x,z,prevFm);
                end

                rng(seed);%to set the same dropout each time..
                aW = feedForward(layers, input, 0);
                if (layers{k}.properties.type==1)
                    layers{k}.fcweight(y,x) = layers{k}.fcweight(y,x) + dw;
                else
                    layers{k}.weight{fm}(y,x,z,prevFm) = layers{k}.weight{fm}(y,x,z,prevFm) + dw;
                    layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (layers{k}.properties.sizeFmInPrevLayer+2*layers{k}.properties.pad));
                end
                                
                rng(seed);%to set the same dropout each time..
                aWPlusDw =  feedForward(layers, input, 0);
                if (layers{k}.properties.type==1)
                    layers{k}.fcweight(y,x) = layers{k}.fcweight(y,x) - dw;
                else
                    layers{k}.weight{fm}(y,x,z,prevFm) = layers{k}.weight{fm}(y,x,z,prevFm) - dw;
                    layers{k}.weightFFT{fm}(:,:,:,prevFm) = fftn( flip(flip(flip(layers{k}.weight{fm}(:,:,:,prevFm),1),2),3) , (layers{k}.properties.sizeFmInPrevLayer+2*layers{k}.properties.pad));
                end
                   
                if (errorMethod==1)
                    cW = -sum((expectedOut).*log(aW{2,end}) + (1-expectedOut).*log(1-aW{2,end}));
                    cWPlusDw = -sum((expectedOut).*log(aWPlusDw{2,end}) + (1-expectedOut).*log(1-aWPlusDw{2,end}));
                else
                    cW       = 0.5*sum((aW{2,end}-expectedOut).^2);
                    cWPlusDw = 0.5*sum((aWPlusDw{2,end}-expectedOut).^2);
                end
                

                estimatedDcDw = (cWPlusDw-cW)/dw;
                diff = estimatedDcDw-calculatedDcDw;
                if ( abs(diff) > th)
                    if ( k == size(layers,2))
                        assert(0,'Big big blunder!!');
                    end
                    assert(0,'How come?? , problem in layer %d',k);
                end
            end
        end
    end
end

%check bias weight
for k=1:size(layers,2)
    if (layers{k}.properties.type==1) % is fully connected layer
        continue; %tested already in upper section
    end
    
    for fm=1:layers{k}.properties.numFm

        calculatedDcDw = layers{k}.bias(fm) - WdCdW{k}.bias(fm);
        rng(seed);%to set the same dropout each time..
        aW       =  feedForward(layers, input, 0);
        layers{k}.bias(fm) = layers{k}.bias(fm) + dw;
        rng(seed);%to set the same dropout each time..
        aWPlusDw =  feedForward(layers, input, 0);
        layers{k}.bias(fm) = layers{k}.bias(fm) - dw;

        if (errorMethod==1)
            cW = -sum((expectedOut).*log(aW{2,end}) + (1-expectedOut).*log(1-aW{2,end})); 
            cWPlusDw = -sum((expectedOut).*log(aWPlusDw{2,end}) + (1-expectedOut).*log(1-aWPlusDw{2,end})); 
        else
            cW       = 0.5*sum((aW{2,end}-expectedOut).^2);
            cWPlusDw = 0.5*sum((aWPlusDw{2,end}-expectedOut).^2);
        end

        estimatedDcDw = (cWPlusDw-cW)/dw;

        diff = estimatedDcDw-calculatedDcDw;
        if ( abs(diff) > th)
            if ( k == size(layers,2) )
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

