function [ answer ] = MSE_Cost( activation,expectedOut )
	answer= 0.5*sum((activation-expectedOut).^2,1);
end