function[SSE,RMS] = erro(ny,nym)

%Calculates RMS and SSE
%INPUTS: 
    %ny is the desired output for the dataset
    %nym is the predicted output

SSE = 0.5*sum((ny-nym).^2);  %Sum of Squares Error
RMS = sqrt(2*SSE/length(ny)); %Root Mean Square Error

end