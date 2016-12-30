function [answer] = dSigmoid(x)
answer=Sigmoid(x).*(1-Sigmoid(x));
end