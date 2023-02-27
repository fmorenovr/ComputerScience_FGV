function [X_train, y_train, X_test, y_test] = read_data(namefile, test_size)
    arguments
        namefile = "car";
        test_size = 0.1;
    end

    if namefile=="car"
        data = readtable("data/Car.csv");
    elseif namefile=="concrete"
        data = readtable("data/Concrete.csv");
    elseif namefile == "raisin_train"
        data = readtable("data/Raisin_train.csv");
    elseif namefile == "raisin_test"
        data = readtable("data/Raisin_test.csv");
    end

    cv = cvpartition(size(data,1),'HoldOut',test_size);
    idx = cv.test;

    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);

    X_train = dataTrain(:,1:end-1);
    X_train = X_train{:,:};
    y_train = dataTrain(:,end);
    y_train = y_train{:,:};
    %data=data(1,:);
    %data=table2cell(data); 

    X_test = dataTest(:,1:end-1);
    X_test = X_test{:,:};
    y_test = dataTest(:,end);
    y_test = y_test{:,:};

end