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

total = 0
for i = 1:m
  h = sigmoid( X(i,:) * theta)
  total = total + (-y(i) * log(h) - (1 - y(i)) * log(1 - h))    
endfor

J = total / m;

no_of_independent_vars = length(X(1,:))
for i = 1:no_of_independent_vars
  total = 0
  for j = 1 : m
    xj = X(j,:)
    h = sigmoid( xj * theta)
    total = total + (h - y(j)) * xj(i)
  endfor
  grad(i) = total/m
endfor
% =============================================================

end
