function [ activations ] = forwardPass( actFunc, x, W, b )
%forwardPass computes the activations for each layer

    numLayers = size(W, 2);
    activations = cell(1, numLayers);
    
    for i = 1:numLayers
        activations{i} = zeros(1, size(W{i}, 2));
    end
    
    activations{1} = actFunc(x * W{1} + repmat(b{1}, size(x,1), 1));
    
    for i = 2:numLayers
        activations{i} = actFunc(activations{i - 1} * W{i} + repmat(b{i}, size(x,1), 1));
    end
end

