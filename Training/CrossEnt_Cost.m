function [ answer ] = CrossEnt_dCost( activation,expectedOut )
	answer=-sum((expectedOut).*log(max(eps,activation)) + (1-expectedOut).*log(max(eps,1-activation)));
end