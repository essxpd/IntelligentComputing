close all;

load('data.mat');

Xtrain = X;
Ytrain = Y;
Xtest = X(80:end,1);
Ytest = Y(80:end,1);

NumNeurons = [1 2 4 8 16 32];
a = 10; 
numTrain = 8; 

figure;
for i = 1:length(NumNeurons)
    %train
    [ W1{i}, W2{i} ] = train_onelayer_batch( Xtrain(1:numTrain), Ytrain(1:numTrain) , NumNeurons(i) );
    
    %test on train
    V1 = W1{i}*horzcat(Xtrain(1:numTrain), ones(size(Xtrain(1:numTrain),1), 1))';
    Y1 = 1./(1 + exp(-a.*(V1))); 
    Y2 = W2{i}'*vertcat(Y1, ones(1,size(Y1,2)));
    errorTrain = (Ytrain(1:numTrain) - Y2').^2; 
    sqErrorTrain(i) = sum(errorTrain);
    
    
    %test
    V1 = W1{i}*horzcat(Xtest, ones(size(Xtest,1), 1))';
    Y1 = 1./(1 + exp(-a.*V1)); 
    Y2 = W2{i}'*vertcat(Y1, ones(1,size(Y1,2)));
    errorTest = (Ytest - Y2').^2; 
    sqErrorTest(i)  = sum(errorTest);
    
    %plot
    subplot(ceil(length(NumNeurons)/2), 2, i); scatter(Xtrain(1:numTrain), Ytrain(1:numTrain), 50, errorTrain, 'x');
    hold on; scatter(Xtest, Ytest, 50, errorTest, 'filled');
    xx = [0:.01:1];
    vv1 = W1{i}*vertcat(xx, ones(1, size(xx,2)));
    yy1 = 1./(1 + exp(-a.*vv1));  
    yy2 = W2{i}'*vertcat(yy1, ones(1, size(xx,2)));
    plot(xx,yy2); title(['(', num2str(i), ') SqErrorTrain: ', num2str(sqErrorTrain(i)), ' SqErrorTest: ', num2str(sqErrorTest(i))]); hold off;
end

figure;
plot(NumNeurons, sqErrorTrain); hold on;
plot(NumNeurons, sqErrorTest);
legend('Training Error', 'Test Error');
