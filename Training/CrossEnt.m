function [ answer ] = CrossEnt( activation,expectedOut )
	answer=(activation-expectedOut)./(activation.*(1-activation));%cross entropy
	
	nanPos = (activation-expectedOut==0) & (activation.*(1-activation)==0); % eliminate 0/0
	answer(nanPos) = 1./(1-2*activation(nanPos));% f/g is undefined when f=0 and g=0 so return the limit f'/g' instead 
end