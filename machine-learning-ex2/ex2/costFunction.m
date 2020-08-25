function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%J = (1/(2*m))*sum((X*theta - y).^2);

cost_J = (1/m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log(1 - sigmoid(X*theta)));
%costJ = 0;

J = cost_J;

grad = zeros(size(theta));

for j= 1:length(grad)
  grad(j) = (1/m) * sum( (sigmoid(X*theta) - y).*X(:,j));
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
