%% Create artificial data
seednum = 4; rng(seednum);
% Sample from a helix
% x = a.cos(a2.t)
% y = a.sin(a2.t)
% z = a.t
max_em_iter = 6;
n = 300;       % Number of 'training' data
V = 2;          % Number of views
% Set the 'speed' of the helix w/ curveparam
% Suggested/interesting values: 1, 4, 10, 20
curveparam = 20; 
% Set the angle of the helix axis over the xy plane. 0 is an 'upright' helix
% Suggested/interesting values: 0, .2, 1
angleparam = .2;
[trainingpoints, G, testpoints] = getartificialGS(n, round(.1*n), curveparam, angleparam);
scatter3(trainingpoints(1, :), trainingpoints(2, :), trainingpoints(3, :)); title('Test-case manifold'); axis([-1 1 -1 1 -1 1]);pause(1);
%% Partition of the data into two views
trainingset = cell(1, V);
trainingset{1} = trainingpoints(1, :);     % This corresponds to the 'sint' component of the helix 
trainingset{2} = trainingpoints(2:3, :);   % This corresponds to the 'cost' and 't' component of the helix 
%
testset = cell(1, V);
testset{1} = testpoints(1, :);
testset{2} = testpoints(2:3, :);
%%% Inspect the training set with
% scatter(trainingset{1}, trainingset{2}(1, :))
% scatter(trainingset{1}, trainingset{2}(2, :))
% Prepare input. Pick specific or all views for training
pickviews = [1 2];
Yin = cell(1, numel(pickviews));
Yin_test = cell(1, numel(pickviews));
for i = 1:numel(pickviews)
    Yin{i} = trainingset{pickviews(i)};
    Yin_test{i} = testset{pickviews(i)};
end
% Train model
max_dx = 1;
for dx = 1:max_dx
fprintf('### dx=%d ###\n', dx);
% options to lllvm_1ep. Include initializations
% Most options are optional. See lllvm_1ep file directly for possible options.
op = struct();
op.seed = 6;
op.max_em_iter = max_em_iter;
% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
op.abs_tol = 1e-5; %originally 1e-3
op.G = G;
op.dx = dx;
% The factor to be added to the prior covariance of C and likelihood. This must
% be positive and typically small.
op.epsilon = 1e-3;
% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
op.alpha0 = 1;
% initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
op.gamma0 = 1;
% Training
[model{dx}, op] = lllvm_1ep(Yin, op, true); %lh 
%load(fname);
end
% Visualize how the estimated (assumed to be univariate) parameter (X) correlates
scatter(model{1}.mean_x, trainingset{2}(2, :));
% TEST - Compute missing view 1 (one-dimensional) given view 2 (two-dimensional)
% After training, use out-of-sample. Here with a set with missing views
Yin_test_missingviews = Yin_test;
Yin_test_missingviews{1} = [];
%Yin_test_missingviews{2} = [];
% Compute out-of-sample and estimate missing views for test
for trainedmodel_dx = 1
    fprintf('##dx=%d##\n', trainedmodel_dx);
    [mean_x_new{trainedmodel_dx}, cov_x_new{trainedmodel_dx}, mean_y_new{trainedmodel_dx}, cov_y_new{trainedmodel_dx}] = ...
    estimatePosteriors(Yin_test_missingviews, model{trainedmodel_dx}, Yin, true);
end
% Check result
scatter(Yin_test{1}', mean_y_new{1}{1}') %These two should be almost equal, ie the graph should be y=x+e
fprintf('Actual (approximated) mean for view 1: %.1f(%.1f)\n', mean(Yin_test{1}'), mean(mean_y_new{1}{1}'));
fprintf('Actual (approximated) st.deviation for view 1: %.1f(%.1f)\n', std(Yin_test{1}'), std(mean_y_new{1}{1}'));
%
co = corrcoef(Yin_test{1}, mean_y_new{1}{1});
for i = 1:numel(Yin_test{1})
    fprintf('Point %d : Posterior mean %+.2f (true value: %+.2f)\n', i, mean_y_new{1}{1}(i), Yin_test{1}(i));
end
fprintf('Correlation coefficient for estimates(posterior means) vs true values: %.3f\n', co(1,2));