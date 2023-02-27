function [J, grad, J_history, it, Theta1, Theta2] = Train_MLP(X, y, Theta1, Theta2, error_stop, alpha, batch_size, lambda, regularization)

    arguments
        X;
        y;
        Theta1;
        Theta2;
        error_stop = 1e-9;
        alpha = 0.01;
        batch_size = 4;
        lambda=1;
        regularization=false;
    end

m = size(X, 1);
[num_labels,k_] = size(unique(y));
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
y_new = zeros(num_labels, m); % 10*5000
for i=1:m
  y_new(y(i)+1,i)=1;
end

X = [ones(m,1) X];
J = nnComputeCost(X, y, y_new, Theta1, Theta2, lambda,regularization);


% Back propagation
i = 1;
it = 0;
J_history = [];
error_rate_change =10; %initialize the error rate
%while (error_rate_change>error_stop) %put the stop condition
for t=1:1000

    % Prediction
    pred = predict(Theta1, Theta2, X);
    J_history = [J_history; nnComputeCost(X, y, y_new, Theta1, Theta2, lambda,regularization)];

    % Step 1
	%a1 = X(batch_size*i+1:batch_size*(i+1),:);
    a1 = X(i,:);
    a1 = a1';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
    
    a2 = [1 ; a2]; % adding a bias
	z3 = Theta2 * a2;
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 2
	delta_3 = a3 - y_new(:,i);
	
    z2=[1; z2];
    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

    % Step 4
	delta_2 = delta_2(2:end);

	Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1';

    % Step 5
    Theta2_grad = (1/m) * Theta2_grad;
    Theta1_grad = (1/m) * Theta1_grad;
    
    % Regularization
    
    if regularization
        Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
        Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
    else
        Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
        Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
    end

    % Update theta
    new_theta1 = Theta1 - alpha*Theta1_grad; %the actualization of the weigths given the gradient
    Theta1 = new_theta1;

    new_theta2 = Theta2 - alpha*Theta2_grad;
    Theta2 = new_theta2;
    
    it=it+1;
    if (it>1)
        error_rate_change = (abs(J_history(it)-J_history(it-1)))/J_history(it-1); %rate of change in the error function
    end
    i= i+1;
    if (i>m)
        i=1;
    end
    
end;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end