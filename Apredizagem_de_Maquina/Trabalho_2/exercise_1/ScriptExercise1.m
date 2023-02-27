clear all;
close all;
clc;


test_size=0.1;

[X_train, y_train, X_test, y_test] = read_data("concrete",test_size);

[m,n] = size(X_test);
X_bias = [ones(m,1), X_test];

[theta, J_history, iterations] = LinearRegressionWD(X_train, y_train);

% plotting the cost function
plot(1: iterations, J_history, '-b');
title("loss history")
xlabel('iterations')
ylabel('loss')

fprintf('Test Error:\t%f\n', erro(y_test, X_bias * theta));
fprintf("Pearson correlation: %d %d\n", corrcoef(y_test, X_bias * theta));