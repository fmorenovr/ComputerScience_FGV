function gaussian()
    x = unifrnd(0,15,1000,1);
    rand_nums = unifrnd(-2,2,1000,1);

    y = sin(2*x) + 1/2*x + 4;
    z = y + rand_nums;

    mean_x = (x-mean(x));
    mean_y = y - mean(y) + rand_nums;

    % least square
    A = [mean_x ones(size(mean_x))];
    c = (A'*A)\(A'*mean_y);

    new_x= linspace(-5,10, 1000);
    new_y = c(1)*new_x+c(2);

    hold on
    plot(x,y, ".", "Color","#FFFF00")
    plot(x,z, ".", "Color","r")
    plot(new_x,new_y, "LineStyle","-", "Color","g", "LineWidth",2)
    plot(new_x+mean(x), new_y+mean(y), "LineStyle","-", "Color","#FFFF00", "LineWidth",2)
    plot(mean_x,mean_y, ".", "Color","#00FFFF")
    set(gca,'Color','k')
    grid on
end