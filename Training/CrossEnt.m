function [ answer ] = CrossEnt( activation,expectedOut )
    assert(~any(activation(:)>1)&&~any(activation(:)<0),'Error when using cross ent, activation must be [0..1], add softmax or change last layer activation');
	answer=(activation-expectedOut)./max(eps,activation.*(1-activation));%cross entropy
	
	nanPos = (activation-expectedOut==0) & (activation.*(1-activation)==0); % eliminate 0/0
	answer(nanPos) = 1./(1-2*activation(nanPos));% f/g is undefined when f=0 and g=0 so return the limit f'/g' instead 
end