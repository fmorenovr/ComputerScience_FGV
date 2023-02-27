function [Theta1, Theta2, J_history, iterations] = create_MLP(X_train, y_train, hidden_layer_size, learning_rate, error_stop, batch_size,regularization)

    arguments
        X_train;
        y_train;
        hidden_layer_size = 5;
        learning_rate = 0.01;
        error_stop = 1e-5;
        batch_size = 4;
        regularization=false;
    end

    [m,n] = size(X_train);
    [m_,k_] = size(unique(y_train));

    input_layer_size  = n;
    num_labels = m_;

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    lambda = 1;

    [J, grad, J_history, iterations, Theta1, Theta2] = Train_MLP(X_train, y_train, ...
                                                                initial_Theta1, initial_Theta2, error_stop, ...
                                                                 learning_rate, batch_size, lambda, regularization);

    %Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
     %            hidden_layer_size, (input_layer_size + 1));

    %Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
      %           num_labels, (hidden_layer_size + 1));    
end