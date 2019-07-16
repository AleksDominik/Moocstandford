function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
dim=8;
Cmat=zeros(dim);
sigmamat=zeros(dim);
erromat=zeros(dim,dim);
for i=0:dim-1
  CMat(i+1)=0.01*(3^i);
  sigmamat(i+1)=0.01*(3^i);
endfor

for j=1:dim
  for k=1:dim
    model=svmTrain(X, y, CMat(j), @(x1, x2) gaussianKernel(x1, x2, sigmamat(k)));
    predictions=svmPredict(model,Xval);
    errormat(j,k)=mean(double(predictions ~= yval));
    %[j,k]
    
  endfor
endfor
[minval, rowmin] = min(min(errormat,[],2));
[minval, colmnmin] = min(min(errormat,[],1));
C=CMat(rowmin);
sigma=sigmamat(colmnmin);








% =========================================================================

end
