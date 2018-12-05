function [answer] = dSigmoid(x)
answer=max(eps,Sigmoid(x).*(1-Sigmoid(x)));
end