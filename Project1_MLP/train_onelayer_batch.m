function [ W1, W2 ] = train_onelayer_batch( X, Y , NumNeurons )

%Set parameters
if(nargin < 3)
    NumNeurons = 4;
end
learningRate = .1;
maxIter = 500000;
stopThresh = 1e-9;
a = 10;
beta = 0.3;

%Initialize 
[numData, numDim] = size(X);

W1 = rand(NumNeurons, numDim+1) / sqrt(numDim);
W2 = rand(NumNeurons + 1, 1);

X = horzcat(X, ones(numData, 1))';
Xr = zeros(numData, NumNeurons, numDim+1);
for i = 1:numData
    Xr(i,:,:) = repmat(X(:,i)', [NumNeurons,1]);
end
stopFlag = 1;
iter = 1;
prevDelta2 = 0;
prevDelta1 = 0;

while(iter < maxIter && stopFlag)
    
    iter = iter + 1;
    
    if(mod(iter, 10000) == 0)
        if(iter > 100000)
            learningRate = .05;
        end
        if(iter > 300000)
            learningRate = .01;
        end
        
        disp(['Iteration: ', num2str(iter), ' Learning Rate: ', num2str(learningRate)]);
    end
    
    
    V1 = W1*X;
    Y1 = 1./(1 + exp(-a*V1));
    Y2 = W2'*vertcat(Y1, ones(1,size(Y1,2)));
    
    err = Y - Y2';
    d_out = err;
    d_in = a*repmat(d_out', [NumNeurons, 1]).*repmat(W2(1:NumNeurons), [1, numData]).*Y1.*(1-Y1);
    
    delta2 = learningRate*repmat(d_out, [1, NumNeurons+1])'.*vertcat(Y1, ones(1,numData));
    delta1 = learningRate*repmat(d_in', [1, 1, numDim+1]).*Xr;

    mDelta2 = (1/numData)*sum(delta2,2);
    delta2 = beta*prevDelta2 + mDelta2;
    
    mDelta1 = reshape(1/numData*sum(delta1), [NumNeurons, numDim+1]);
    delta1 = beta*prevDelta1 + mDelta1;  
    
    prevDelta2 = delta2; 
    prevDelta1 = delta1;
    
    W2 = W2 + delta2;
    W1 = W1 + delta1;    

    if(sum(sum(abs(delta1))) + sum(sum(abs((delta2)))) < stopThresh)
        stopFlag = 0;
    end
    
    
end
end

