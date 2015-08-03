function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% input layer
a_1 = [ones(m,1) X];
% hidden layer
a_2 = [ones(m, 1) sigmoid(a_1 * Theta1')];
% output layer
H_x = sigmoid(a_2 * Theta2');

y_mod = zeros(size(H_x));
% set the desired value to 1 and rest to 0
for n = 1:m
    y_mod(n, y(n)) = 1;
end

partA = -y_mod.*log(H_x);
partB = (1-y_mod).*log(1-H_x);
Sum = sum(sum(partA - partB));

theta_1_tmp = Theta1(:, 2:end);
theta_2_tmp = Theta2(:, 2:end);
reg_value = (lambda/(2*m)) * (sum(sum(theta_1_tmp.^2)) + sum(sum(theta_2_tmp.^2)));
J = (Sum/m) + reg_value;

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t = 1:m
	a1_t = a_1(t,:); % 1 * 401
	a2_t = a_2(t,:); % 1 * 26
	a3_t = H_x(t,:); % 1 * 10
	yMod_t = y_mod(t,:); % 1 * 10
	z2_t = [1 a1_t * Theta1']'; % 1 * 26
	
	% Error value for output node
	delta3_t = (a3_t - yMod_t)'; % 10 * 1
	
	% Error value for hidden layer
	delta2_t = Theta2' * delta3_t .* sigmoidGradient(z2_t); % 26 * 1
	
	delta1 = delta1 + delta2_t(2:end) * a1_t; % 25 * 401
	delta2 = delta2 + delta3_t * a2_t; % 10 * 26
end;

Theta1_grad = (1 / m) * delta1 + (lambda/m) * [zeros(size(Theta1, 1), 1) theta_1_tmp];
Theta2_grad = (1 / m) * delta2 + (lambda/m) * [zeros(size(Theta2, 1), 1) theta_2_tmp];


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
