function [J] = nnComputeCost(X, y, y_new, Theta1, Theta2, lambda, regularization)
        
        [num_labels,k_] = size(unique(y));

        m = size(X, 1);
        z2 = Theta1 * X';
        a2 = sigmoid(z2);
        
        a2 = [ones(m,1) a2'];
        z3 = Theta2 * a2';
        h_theta = sigmoid(z3);
        
        J = (1/m) * sum ( sum ( (-y_new) .* log(h_theta) - (1-y_new) .* log(1-h_theta) ));

        if regularization
                t1 = Theta1(:,2:size(Theta1,2));
                t2 = Theta2(:,2:size(Theta2,2));
                Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);
                J = J + Reg;
        end 
end