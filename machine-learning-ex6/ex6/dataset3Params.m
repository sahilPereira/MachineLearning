function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values_list = [0.01 0.03 0.1 0.3 1 3 10 30];
min_error = Inf;

for i = 1:length(values_list)
	for j = 1:length(values_list)
		current_c = values_list(i);
		current_sigma = values_list(j);

		model = svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
		prediction = svmPredict(model, Xval);
		error = mean(double(prediction ~= yval));

		if error < min_error
			min_error = error;
			C = current_c;
			sigma = current_sigma;
		end
	end
end

% =========================================================================

end
