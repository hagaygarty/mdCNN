function [res] = sumDim(input,dims)
    res=input;
    for dim=dims
        res = sum(res,dim);
    end
end
