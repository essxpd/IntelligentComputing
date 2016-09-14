function [ W, b ] = initNetwork( layerSizes, numDimensions )
%initNetwork assigns values to W and b
    
    sizes = [numDimensions, layerSizes];

    min = -(1 / sqrt(numDimensions));
    max = -min;
    
    for i = 1 : size(sizes, 2) - 1
        W{i} = min + (max - min) .* rand(sizes(i), sizes(i + 1));
        b{i} = min + (max - min) .* rand(1, sizes(i + 1));
%         b{i} = zeros(1, sizes(i + 1));
    end
end

