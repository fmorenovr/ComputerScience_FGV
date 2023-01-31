function [mu, sigma, p] = em()
    data = load("exercise5.mat");
    [C0,C1]=deal(data.C0,data.C1);

    % we join all data in one
    K = 2; % number of clusters/classes/Gaussian Mixture components
    CX = [C0;C1];
    D = size(CX,2); % dimension
    N = size(CX,1); % number of samples
    % We use a Keans to get an initial means
    [idx,mu] = kmeans(CX,K);

    % compute the covariance of the components
    sigma = zeros(D,D,K);
    for k = 1:K
        sigma(:,:,k) = cov(CX(idx==k,:));
    end

    % E-Step

    % variables for convergence 
    p = [0.2, 0.3, 0.2, 0.3]; % arbitrary pi
    converged = 0;
    prevLoglikelihood = Inf;
    prevMu = mu;
    prevSigma = sigma;
    prevPi = p;
    round = 0;
    while (converged ~= 1)
        round = round +1;
        gm = zeros(K,N); % gaussian component in the nominator
        sumGM = zeros(N,1); % denominator of responsibilities
        % E-step:  Evaluate the responsibilities using the current parameters
        % compute the nominator and denominator of the responsibilities
        for k = 1:K
            for i = 1:N
                 Xmu = CX-mu(k,:);
                 logPdf = log(1/sqrt(det(sigma(:,:,k))*(2*pi)^D));% + (-0.5*(sigma(:,:,k)\Xmu')*Xmu);
                 gm(k,i) = log(p(k)) * logPdf;
                 sumGM(i) = sumGM(i) + gm(k,i);
             end
        end
    
        % calculate responsibilities
        res = zeros(K,N); % responsibilities
        Nk = zeros(4,1);
        for k = 1:K
            for i = 1:N
                % I tried to use the exp(gm(k,i)/sumGM(i)) to compute res but this leads to sum(pi) > 1.
                res(k,i) = gm(k,i)/sumGM(i);
            end
            Nk(k) = sum(res(k,:));
        end

    % M-Step
    for k = 1:K
        for i = 1:N
            mu(k,:) = mu(k,:) + res(k,i).*CX(k,:);            
        end
        mu(k,:) = mu(k,:)./Nk(k);
    
        for i = 1:N
            sigma(:,:,k) = sigma(:,:,k) + res(k,i).*(CX(k,:)-mu(k,:))*(CX(k,:)-mu(k,:))';
        end
        mu(k,:) = mu(k,:)./Nk(k);
        sigma(:,:,k) = sigma(:,:,k)./Nk(k);
        p(k) = Nk(k)/N;
    end
end