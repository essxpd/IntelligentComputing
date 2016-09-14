function [ p ] = predict( actFunc, input, W, b )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here

    output = forwardPass(actFunc, input, W, b);
    [~, p] = max(output{size(W,2)}, [], 2);
end

