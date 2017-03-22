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

predictions = sigmoid(X * theta);

%reg_term = (lambda / (2 * m)) * sum(theta .* theta);

reg_term = 0;
for i = 2 : size(theta),
	reg_term = reg_term + theta(i) * theta(i);
end
reg_term = (lambda / (2 * m)) * reg_term;

delta = (-y' * log(predictions) - (1 - y') * log(1 - predictions));
J = (1 / m) * delta + reg_term;

grad = ((1 / m) * X' * (predictions - y)) + (lambda / m) * theta;

grad(1) = (1 / m) * X(:,1)' * (predictions - y);
%grad = (1 / m) * X' * (predictions - y);

% =============================================================

end
