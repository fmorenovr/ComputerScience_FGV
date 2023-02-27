function sinoide(n,D)
    arguments
        n = 100
        D = 3
    end
    x = sort(unifrnd(0,1,n,1));
    epsilon = normrnd(0,0.3,n,1);

    y = sin(2*pi*x);
    z = y + epsilon;
    
    A = [x.^3 x.^2 x ones(size(x))];
    c = (A'*A)\(A'*z);
    c2 = A\z;

    y_p = c(1)*x.^3 + c(2)*x.^2 + c(3)*x + c(4);

    disp(c)
    disp(c2)
    %poly = polyfit(x,y,D);
    %y_p = polyval(poly, x);
    hold on
    plot(x,y, "-", "Color","g", LineWidth=2)
    plot(x,z, "o", "Color","b")
    plot(x,y_p, "-", "Color","r", LineWidth=2)
    grid on
end
