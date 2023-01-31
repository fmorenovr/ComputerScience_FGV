function plot_2d()
    a = unifrnd(30,40,1000,1);
    b = unifrnd(10,20,1000,1);
    uniform_d = [a b];

    mu = [20 35];
    Sigma = [1 0; 0 1];
    normal_d = mvnrnd(mu,Sigma,1000);

    hold on

    plot(uniform_d(:,1),uniform_d(:,2),'o', "Color","b", 'DisplayName','Uniform Distribution')
    plot(normal_d(:,1),normal_d(:,2),'*', "Color","r", 'DisplayName','Normal Distribution')
    xlim([0 50])
    ylim([0 50])
    xlabel('x axis')
    ylabel('y axis')
    legend
    hold off
end