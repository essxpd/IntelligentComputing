function [ y ] = sigmoid_grad( x )
% Sigmoid function gradient
    y = (1 - sigmoid(x)) .* sigmoid(x);
end

