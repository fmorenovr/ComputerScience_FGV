function distance_matrix = knn(k)
    arguments
       k=3
    end

    load fisheriris.mat

    sepals = [meas(:,1) meas(:,2)];
    petals = [meas(:,3) meas(:,4)];

    figure(1);
    gscatter(sepals(:,1), sepals(:,2), species,'rgb','osd');
    xlabel('Sepal length');
    ylabel('Sepal width');

    figure(2);
    gscatter(petals(:,1), petals(:,2), species,'rgb','osd');
    xlabel('Petal length');
    ylabel('Petal width');

    figure(3);
    gscatter(petals(:,1), sepals(:,1), species,'rgb','osd');
    xlabel('Petal length');
    ylabel('Sepal length');

    figure(4);
    gscatter(petals(:,2), sepals(:,2), species,'rgb','osd');
    xlabel('Petal width');
    ylabel('Sepal width');

    figure(5);
    gscatter(sepals(:,1).*sepals(:,2), meas(:,1).*meas(:,2), species,'rgb','osd');
    xlabel('Sepal Area');
    ylabel('Petal Area');

    % select the best
    distance_matrix = squareform(pdist([petals(:,1) petals(:,2)]));
end
