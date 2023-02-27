function [J, grad] = ComputeCost(i, X, y, theta, lambda)
    % Prepare Variables
    [m, n] = size(X);
    
    % Calculate Hypothesis
    h = X * theta;
    
    % Calculate Cost
    J_partial = 1 / (2 * m) * sum((h - y) .^ 2);
    J_regularization = (lambda/(2*m)) * sum(theta(2:end).^2);
    J = J_partial + J_regularization;

    % claculate grad
    %Grad without regularization
    grad_partial = (1/m) * (h(i) -y(i)*X(i,:)' );

    %%Grad Cost Added
    %grad_regularization = zeros(n, 1);
    %grad_regularization(i) = 2*(lambda/m) * theta(i);
    grad_regularization = (lambda/m) .* theta(2:end);
    grad_regularization = [0; grad_regularization];

    grad = grad_partial + grad_regularization;
end