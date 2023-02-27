function [theta, J_history, it] = LinearRegressionWD(X_train, y_train, alpha,lambda, error_stop)
    arguments
        X_train
        y_train
        alpha = 0.01;
        lambda = 0.001;
        error_stop = 1e-5;
    end

    
    [m,n] = size(X_train);
    X_bias = [ones(m,1), X_train];

    theta = 0.01*randn(n+1,1); %initialize the weigth
    error_rate_change =10; %initialize the error rate

    i = 1;
    it = 0;
    J_history = [];

    while (error_rate_change>error_stop) %put the stop condition
        pred = X_bias * theta;
        
        [J, grad] = ComputeCost(i, X_bias, y_train, theta, lambda);

        J_history = [J_history; sum(J)]; %compute the cross entropy error and saved it in a vector
        new_theta = theta - alpha*grad; %the actualization of the weigths given the gradient
        
        %new_theta = theta - (alpha * (1 / m) * sum((pred - y) .* X_bias(:, i)));

        theta = new_theta;
        it=it+1;
        if (it>1)
            error_rate_change = (abs(J_history(it)-J_history(it-1)))/J_history(it-1); %rate of change in the error function
        end
        i= i+1;
        if (i>m)
            i=1;
        end
    end
end