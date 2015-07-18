function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H_x = sigmoid(X * theta);
log_Hx = log(H_x);
log_small_Hx = log(1-H_x);
Inner_exp = (-y.*log_Hx - (1-y).*log_small_Hx);
Sum = sum(Inner_exp)/m;

altTheta = theta;
altTheta(1) = 0;

regSum = sum(altTheta.^2)*(lambda/(2*m));
J = Sum+regSum;

grad = (X' * (H_x - y)) * (1/m) + altTheta*(lambda/m);


% =============================================================

end
