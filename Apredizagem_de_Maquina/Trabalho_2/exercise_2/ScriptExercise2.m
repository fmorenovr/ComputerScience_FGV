clear all;
close all;
clc;

[X_train, y_train] = read_data("raisin_train");
[X_test, y_test] = read_data("raisin_test");

y_train = categorical(y_train);
[y_train_labels, ~, y_train] = unique(y_train);

y_test= categorical(y_test);
[y_test_labels, ~, y_test] = unique(y_test);

[Theta1, Theta2, J_history, iterations] = create_MLP(X_train, y_train-1);

m = size(X_test, 1);
X_test = [ones(m,1) X_test];
y_pred = predict(Theta1, Theta2, X_test);
    
fprintf('Test Error:\t%f\n', erro(y_test-1, y_pred-1));

 % plotting the cost function
 figure(1);
plot(1: iterations, J_history, '-b');
title("loss history")
xlabel('iterations')
ylabel('loss')

figure(2);
plotroc(y_test',y_pred');