function [Xtrain, Gtrain, Xtest] = getartificialGS(Ntrain, Ntest, a, b)
% Sample off a 1-D manifold.
%
% The training set will be uniformly sampled on the manifold,
%   making sure that the neighbourhood matrix is 'ideal'.
% The test set consists of random samples across the manifold.
% Ntrain, Ntest:    Number of data required
% Xtrain, Xtest:    Resulting samples
% Gtrain:           Neighbourhood matrix for training set samples
%
% Examples:
%   getartificialGS(100, 10, 5);    % A helix w/ few revolutions
%   getartificialGS(100, 10, 1);    % A hyperbolic-like curve
%   getartificialGS(2000, 10, 50);  % Fast-revolving helix
% Visualise output with:
%   scatter3(Xtrain(1, :), Xtrain(2, :), Xtrain(3, :))
%
% GS 2017
ztrain = 1:Ntrain;
ztrain = ztrain - mean(ztrain); ztrain = ztrain / max(ztrain);
ztest = rand(1, Ntest)*2 - 1;
xtrain = cos(a*ztrain) + b*ztrain; xtest = cos(a*ztest) + b*ztest;
ytrain = sin(a*ztrain); ytest = sin(a*ztest);
Xtrain = [xtrain; ytrain; ztrain];
Xtest =  [xtest; ytest; ztest];
% Add noise
Xtrain = Xtrain + .005*randn(size(Xtrain));
Gtrain = zeros(Ntrain, Ntrain);
for i = 1:Ntrain-1
    Gtrain(i, i+1) = 1;
    Gtrain(i+1, i) = 1;
end
return;