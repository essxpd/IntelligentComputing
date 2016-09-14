clear all

% Load the data in
data_file = 'semeion.data';
% data_file = 'wine.data';
load(data_file);

if strcmp(data_file, 'threeclouds.data')
    data = threeclouds;
elseif strcmp(data_file, 'wine.data')
    data = wine;
elseif strcmp(data_file, 'semeion.data')
    data = semeion; 
end

% Shuffle the data
data = data(randperm(size(data,1)),:);

% Seperate into network input data and labels
labels = data(:,1);
input = data(:, 2:size(data,2));

% Clean the input
for col = 1:size(input,2)
    
     input(:,col) = (input(:,col) - mean(input(:,col),1)) ./ std(input(:,col),0,1);
end

% Determine dimensionality, number of classes, and number of examples
numDimensions = size(2:size(data,2), 2);
numClasses = size(unique(data(:,1)), 1);
numExamples = size(data(:,1), 1);

% Create the one hot expected output
expectedOutput = zeros(size(2:size(data,2),1), numClasses);
for i = 1:size(labels, 1)
    expectedOutput(i, labels(i)) = 1;
end

% Divide into Training, Validation, and Test sets
ssizes = [.8, .1, .1];  % Set sizes
c = cumsum(ssizes);     % Cummulative sume
trainingData = input(1:floor(numExamples*c(1)), :);
trainingLabels = labels(1:floor(numExamples*c(1)), :);
trainingExpectedOutput = expectedOutput(1:floor(numExamples*c(1)), :);

validationData = input(floor(numExamples*c(1))+1:floor(numExamples*c(2)), :);
validationLabels = labels(floor(numExamples*c(1))+1:floor(numExamples*c(2)), :);
validationExpectedOutput = expectedOutput(floor(numExamples*c(1))+1:floor(numExamples*c(2)), :);

testData = input(floor(numExamples*c(2))+1:numExamples, :);
testLabels = labels(floor(numExamples*c(2))+1:numExamples, :);
testExpectedOutput = expectedOutput(floor(numExamples*c(2))+1:numExamples, :);

% If 2d or 3d then plot the points with a scatter
if numDimensions == 2
    scatter(input(:,1)', input(:,2)', 15, labels', 'filled');
elseif numDimensions == 3
    scatter3d(input(:,1)', input(:,2)', input(:,3), 15, labels', 'filled');
end

% Network Config
eta = .1;
maxEpoch = 10;
actFunc = @sigmoid;
actFuncGrad = @sigmoid_grad;

layerSizes = [14, numClasses]; % Output layer is one hot 
numHiddenLayers = size(layerSizes, 2) - 1;
outputLayer = size(layerSizes, 2);

% Initialize the network
% rng(0);
[W, b] = initNetwork(layerSizes, numDimensions);

% Train the network
% for k = 1:1
%     in = input(:,k);
%     
%     activations = forwardPass(actFunc, in, W, b);
% end

% activations = forwardPass(actFunc, input(1,:), W, b);
% W{1}(1, 1) = .1;
% W{1}(2, 1) = .1;
% W{1}(1, 2) = .25;
% W{1}(2, 2) = .7;
% W{2}(1, 1) = .4;
% W{2}(2, 1) = .6;
% W{2}(1, 2) = .5;
% W{2}(2, 2) = .3;

% test = forwardPass(actFunc, input(2,:), W, b);

s = sprintf('\nEpoch\t|\tValidation accuracy\n---------------------------------');
disp(s);

for e = 1:10
    for k = 1:size(trainingData,1)
        [delta_W, delta_b] = backPropagation(actFunc, actFuncGrad, W, b, trainingData(k,:), trainingExpectedOutput(k,:));

        for i = 1:size(W,2)
            W{i} = W{i} + eta .* delta_W{i};
            b{i} = b{i} + eta .* delta_b{i};
        end
    end
    
    p = predict(actFunc, validationData, W, b);
    numCorrect = numel(find(p == validationLabels));
    s = sprintf('%i\t\t|\t%.2f%%', e, (numCorrect/size(validationLabels,1))*100);
    disp(s);
end

% test = forwardPass(actFunc, input(2,:), W, b);

p = predict(actFunc, testData, W, b);
numCorrect = numel(find(p == testLabels));
s = sprintf('\nTest accuracy: %.2f%%\n', (numCorrect/size(testLabels,1))*100);
disp(s);