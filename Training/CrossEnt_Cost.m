function [ answer ] = CrossEnt_Cost( activation,expectedOut )
	answer=-( expectedOut.*log(max(eps,activation)) + (1-expectedOut).*log(max(eps,1-activation)));
end