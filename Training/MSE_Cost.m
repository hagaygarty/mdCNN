function [ answer ] = MSE_Cost( activation,expectedOut )
	answer= 0.5*(activation-expectedOut).^2;
end