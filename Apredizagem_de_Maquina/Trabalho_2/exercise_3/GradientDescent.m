function [theta, J_history] = GradientDescent(X, y, theta, alpha)
    % Prepare Variables
    m = length(y);
    h = X * theta;
    new_theta = zeros(m,1);

    new_theta(1) = theta(1) - (alpha * (1 / m) * sum(h - y));
    theta(1) = new_theta(1);

    for i = 2 : m,
        new_theta(i) = theta(i) - (alpha * (1 / m) * sum((h - y) .* X(:, i)));
        theta(i) = new_theta(i);
    end
    J_history = ComputeCost(X, y, theta);
end