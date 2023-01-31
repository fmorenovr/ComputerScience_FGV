function plot_3d()
    rand_values = unifrnd(0,1,20000,3);
    m_cond = rand_values >0.9;
    idx = sum(m_cond, 2)>=1;

    random_d = rand_values(idx,:);
    plot3(random_d(:,1), random_d(:,2), random_d(:,3), '*', "Color","b")
    hold on

    m_cond_1 = rand_values < 0.5;
    m_cond_2 = rand_values > 0.4;
    idx_2 = find( sum(m_cond_1,2)==3 & sum(m_cond_2,2)>=1 );
    random_d_2 = rand_values(idx_2,:);
    plot3(random_d_2(:,1), random_d_2(:,2), random_d_2(:,3), '*', "Color","r")

    %plot(uniform_d(:,1),uniform_d(:,2),'o', "Color","b", 'DisplayName','Uniform Distribution')
    %plot(normal_d(:,1),normal_d(:,2),'*', "Color","r", 'DisplayName','Normal Distribution')
    %xlim([0 50])
    %ylim([0 50])
    axis equal
    grid on
    xlabel('x axis')
    ylabel('y axis')
    zlabel('z axis')
    %legend
    hold off
end