function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% H_x need to be a vector, hence we cannot use theta' * X'
% Using X * theta returns a vector, while other way does not
H_x = sigmoid(X * theta);
log_Hx = log(H_x);
log_small_Hx = log(1-H_x);
Inner_exp = (-y.*log_Hx - (1-y).*log_small_Hx);
Sum = sum(Inner_exp);
J = Sum/m;

grad = ((H_x - y)' * X) * (1/m);

% =============================================================

end
