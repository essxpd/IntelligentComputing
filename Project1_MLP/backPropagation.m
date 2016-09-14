function [ delta_W, delta_b ] = backPropagation( actFunc, actFuncGrad, W, b, x, t )
%backPropagation Calclates the error for each layer

    numLayers = size(W, 2);
    activations = cell(1, numLayers);
    z = cell(1, numLayers);
    delta = cell(size(W));
    delta_W = cell(size(W));
    delta_b = cell(size(b));
    
    for i = 1:numLayers
        activations{i} = zeros(1, size(W{i}, 2));
        z{i} = zeros(1, size(W{i}, 2));
        delta{i} = zeros(1, size(W{i}, 2));
        delta_W{i} = zeros(size(W{i}));
        delta_b{i} = zeros(size(b{i}));
    end
    
    
    % Forward pass
    z{1} = x * W{1} + repmat(b{1}, size(x,1), 1);
    activations{1} = actFunc(z{1});
    
    for i = 2:numLayers
        z{i} = activations{i - 1} * W{i} + repmat(b{i}, size(x,1), 1);
        activations{i} = actFunc(z{i});
    end

    % Backward pass
    delta{numLayers} = (t - activations{numLayers}) .* actFuncGrad(z{numLayers});
    delta_W{numLayers} = activations{numLayers - 1}' * delta{numLayers};
    delta_b{numLayers} = delta{numLayers};
    
    for i = numLayers:-1:3
        delta{i - 1} = (delta{i} * W{i}') .* actFuncGrad(z{i - 1});
        delta_W{i - 1} = activations{i - 2}' * delta{i - 1};
        delta_b{i - 1} = delta{i - 1};
    end
    
    delta{1} = (delta{2} * W{2}') .* actFuncGrad(z{1});
    delta_W{1} = x' * delta{1};
    delta_b{1} = delta{1};
end

