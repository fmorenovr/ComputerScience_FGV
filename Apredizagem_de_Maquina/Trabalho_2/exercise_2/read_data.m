function [X, y] = read_data(namefile, test_size)
    arguments
        namefile = "car";
        test_size = 0.1;
    end

    if namefile=="car"
        data = readtable("../data/Car.csv");
    elseif namefile=="concrete"
        data = readtable("../data/Concrete.csv");
    elseif namefile == "raisin_train"
        data = readtable("../data/Raisin_train.csv");
    elseif namefile == "raisin_test"
        data = readtable("../data/Raisin_test.csv");
    end

    X = data(:,1:end-1);
    X = X{:,:};
    y = data(:,end);
    y = y{:,:};

end