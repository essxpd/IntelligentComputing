clear;

%Parameters
w1 = [0.1 0.1]';
w2 = [0.25 0.7]';
w3 = [0.4 0.6]';
w4 = [0.5 0.3]';
x = [1 1]';
t = [1 0]';
eta = 0.1;

%For plot
xx = [-2:0.25:2];
[X, Y] = meshgrid(xx, xx);
Xdata(:,1) = X(:);
Xdata(:,2) = Y(:);

for n=1:5

  % Plot decision surface and x
  % hlo = 1 ./ (1 + e.^(-[w1 w2]'*Xdata'));
  % olo = 1 ./ (1 + e.^(-[w3 w4]'*hlo));
  % figure;
  % subplot(1,2,1); scatter(Xdata(:,1), Xdata(:,2), 10, olo(1,:), 'filled');
  % subplot(1,2,2); scatter(Xdata(:,1), Xdata(:,2), 10, olo(2,:), 'filled'); 
  
  %Forward pass
  v1 = w1'*x;
  o1 = 1 / (1 + exp(-v1));
  v2 = w2'*x;
  o2 = 1 / (1 + exp(-v2));
  
  v3 = w3'*[o1 o2]';
  o3 = 1 / (1 + exp(-v3))
  v4 = w4'*[o1 o2]';
  o4 = 1 / (1 + exp(-v4))

  %Backward pass
  d3 = (t(1) - o3)*o3*(1-o3);
  d4 = (t(2) - o4)*o4*(1-o4);
  w3 = w3 + (eta * d3 * [o1 o2]')
  w4 = w4 + (eta * d4 * [o1 o2]')
  
  d1 = o1*(1-o1)*( d3*w3(1) + d4*w4(1) );
  d2 = o2*(1-o2)*( d3*w3(2) + d4*w4(2) );
  w1 = w1 + (eta * d1 * x)
  w2 = w2 + (eta * d2 * x)

end  
