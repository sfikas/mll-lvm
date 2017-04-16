function [mean_x_new, cov_x_new, mean_y_new, cov_y_new, naive_y_new] = estimatePosteriors(Y_test, model, Y_train, usex)
% Estimate the latent vectors given a set of observed views 
% of a set of unseen data aka "out-of-sample".
%
% Y_new is a cell of observations. Some views may be empty.
% Model is the struct given as the result of lllvm1_ep.
%
% mean_x_new, cov_x_new are the estimate moments of x for the new data.
% mean_y_new, cov_y_new are the estimate moments of y (the missing part) for the new data.
% naive_y_new is the estimate of y based on the mean of its nns.
%
% G.Sfikas June 2016

if exist('usex', 'var') && usex
    % Filter out nns using x estimate
    x_kstart = 40; x_kmin = 3;
end
dx = model.dx;
V = numel(Y_test);
assert(V == numel(model.gamma));
assert(V == numel(model.cov_c));
assert(V == numel(model.mean_c));
assert(V == numel(Y_train));
% Calculate distance of y to the rest
N_train = size(Y_train{1}, 2);
for i = 1:V
    if(isempty(Y_test{i}))
        continue;
    end
    N_test = size(Y_test{i}, 2);
end
Y_traintest = cell(1, V);
for i = 1:V
    if(~isempty(Y_test{i}))
        Y_traintest{i} = cat(2, Y_train{i}, Y_test{i});
    end
end
% Computes D for all set, will only use the distances of the test set
% against the training set though.
% For neighbourhoods of training against training points we'll use model.G.
% We don't want to change the part of the graph that is already known and
% has been for a converged trained model. (that would move us farther from the
% equillibrium unnecessarily)

% Choose neighbours of _test data_ (only) using k nearest neighbours.
k = 5; use_knn = false;
D = computeDistances_multiview(Y_traintest);
if(~use_knn)
    [~, distance_threshold] = print_stats(D, k);
end
D_roi = D((N_train+1):end, 1:N_train);
nn = zeros(size(D_roi));
for n = 1:size(D_roi, 1)
    if(use_knn)
        [~, I] = sort(D_roi(n, :));
        nn(n, I(1:k)) = 1;
        clear I;
    else
        %Use a distance threshold to create the neighbourhood
        nearest_neighbours = D_roi(n, :) < distance_threshold;
        nn(n, nearest_neighbours) = 1;
        if(nnz(nearest_neighbours) == 0)
            %If no data are under the threshold, force-add the
            %nearest datum to the neighbourhood
            nn(n, argmin(D_roi(n, :))) = 1;
        end
    end
end
% After having computed distances, normalize the data
% using the same procedure found in lllvm_1ep
Y_train_unscaled = Y_train;
for i = 1:V
    Y_train{i} = (Y_train{i} - model.norm_minus{i}*ones(1, N_train)) / model.norm_divide{i};
    if(~isempty(Y_test{i}))
        Y_test{i} = (Y_test{i} - model.norm_minus{i}*ones(1, N_test)) / model.norm_divide{i};
    end
end
mean_x_new = zeros(dx, N_test);
cov_x_new = zeros(dx, dx, N_test);
model.Y = Y_train;
model.dy = cell(1, V);
model.N_train = N_train;
for i = 1:V
    model.dy{i} = size(Y_train{i}, 1);
end
% Compute p(x|.) (out of sample) for each test datum
% Also get p(y|.) if there were missing views in the input
mean_y_new = cell(1, V);
cov_y_new = cell(1, V);
naive_y_new = cell(1, V);
for n = 1:N_test
    y_new = cell(1, V);
    for i = 1:V
        if(isempty(Y_test{i}))
            continue;
        end
        y_new{i} = Y_test{i}(:, n);
    end
    if(~exist('x_kstart', 'var'))
        [mean_x_new(:, n), cov_x_new(:, :, n), mean_y_new_allviews, cov_y_new_allviews] = ...
        posteriorX(y_new, model, nn(n, :));        
    else
        [mean_x_new(:, n), cov_x_new(:, :, n), mean_y_new_allviews, cov_y_new_allviews] = ...
        posteriorX(y_new, model, nn(n, :), x_kstart, x_kmin);
    end
    for i = 1:V
        if(isempty(mean_y_new_allviews{i}))
            continue;
        end
        mean_y_new{i}(:, end+1) = mean_y_new_allviews{i}(:);
        cov_y_new{i}(:, end+1) = cov_y_new_allviews{i}(:);
        naive_y_new{i}(:, end+1) = mean(Y_train_unscaled{i}(:, logical(nn(n, :))), 2);
    end
    fprintf('.');
    if(mod(n, 10) == 0)
        fprintf('\n');
    end
end
return;

function [mean_x_output, cov_x_output, mean_y_output, cov_y_output] = posteriorX(y, model, nn, x_kstart, x_kmin)
% Calculates the posterior of x given a single multi-view new observation.
%
% nn keeps track of the nearest neighbours of the new datum.
%
% G Sfikas June 2016
V = numel(model.Y);
%
% Create variables that correpond to the training set parameters + the new
% datum.
%

% Create the full G matrix (training + one new datum)
G = model.G;
G(model.N_train+1, 1:model.N_train) = nn;
G(1:model.N_train, model.N_train+1) = nn;
assert(nnz(G(end, 1:end-1) ~= G(1:end-1, end)') == 0);
% Create the full Y observations
Y = model.Y;
missing_views = false(1, V);
for i = 1:V
    if(~isempty(y{i}))
        Y{i}(:, end+1) = y{i};
    else
        missing_views(i) = true;
    end
end
% mean_y and cov_y have the distribution of the missing views of y, if any
mean_y = cell(1, V);
cov_y = cell(1, V);
% Calculate initial parameters for posterior of new datum
[mean_x_init cov_x_init] = meanparams_x(model, nn);
[mean_c_init cov_c_init] = meanparams_c(model, nn);
% Filter out nearest neighbours according their x- value
if(exist('x_kstart', 'var'))
    nn = filter_nns(model, mean_x_init, nn, x_kstart, x_kmin);
end
% Add the x and c params to the data vector
% Either initial x or c is really unnecessary; E- step will do the rest
[mean_x, cov_x] = addnew_x(model, mean_x_init, cov_x_init);
[mean_c, cov_c] = addnew_c(model, mean_c_init, cov_c_init);
% Create deterministic params
gamma_new = model.gamma;
alpha_new = model.alpha;
epsilon = model.epsilon;
% Create Ltilde
L = diag(sum(G, 1)) - G;
[Ltilde] = compute_Ltilde(L, epsilon, gamma_new, 1);
% Create other params
dx = model.dx;
J = kron(ones(model.N_train+1,1), eye(dx));
invOmega = kron(2*L, eye(dx));
invPi_new = alpha_new*eye((model.N_train + 1)*dx) + invOmega;
% Gamma and H, initialized as empty cells (just so matlab doesnt complain)
Gamma = cell(1, V);
H = cell(1, V);
A = cell(1, V);
b = cell(1, V);
% We'll use the original code to run the Expectation step.
% This will cycle for a number of iterations between estimates of the
% distributions of C and x.
% The distributions that correspond to both training and the test data
% are updated at first, and then the training data distributions are
% reverted to their original state. This may seem counterproductive but it
% is actually much simpler to implement.
%if(false)
debug_on = false;
if(debug_on)
    figure;
    hold off;
    axis normal;
    axis manual;
end
for cycles = 1:2
    % Compute posterior for y in case there are missing views
    for i = 1:V
        if(missing_views(i))
            % Size dy x dy
            % L(model.N_train+1, ...) is equal to
            % the number of neighbours of the new datum
            cov_y{i} = inv(epsilon + 2*gamma_new{i}*L(model.N_train+1, model.N_train+1))*eye(model.dy{i});
            % Size dy
            evalue = 0;
            % Index of new datum
            idx_start_us = model.N_train*dx + 1;           
            for mv_nn = find(nn)
                % Index of neighbour
                idx_start_neighbour = (mv_nn-1)*dx + 1;
                cfactor = mean_c{i}(:, idx_start_us:idx_start_us+dx-1) + ...
                          mean_c{i}(:, idx_start_neighbour:idx_start_neighbour+dx-1);
                xfactor = -mean_x(idx_start_neighbour:idx_start_neighbour+dx-1) ...
                          +mean_x(idx_start_us:idx_start_us+dx-1);
                evalue = evalue + Y{i}(:, mv_nn) + .5*cfactor*xfactor;
            end
            evalue = 2*gamma_new{i}*evalue - .5*epsilon*sum(Y{i}(:, 1:model.N_train), 2);
            mean_y{i} = cov_y{i} * evalue;
            Y{i}(:, model.N_train+1) = mean_y{i};
            %Debug
            %tt = mean(model.Y_nonorm{i}(:, logical(nn)), 2);
            %fprintf('#%d# Sum of unnormalized neighbours: %f\n', i, tt);
            %Y{i}(:, model.N_train+1) = sum(Y{i}(:, logical(nn)), 2);
        end
    end
    % Compute posterior for x
    invcov_x = invPi_new;
    bsum = 0;
    for i = 1:V
        [A{i}, b{i}] = compute_suffstat_A_b(G, mean_c{i}, cov_c{i}, Y{i}, gamma_new{i}, epsilon);
        invcov_x = invcov_x + A{i};
        bsum = bsum + b{i};
    end
    cov_x = inv(invcov_x);
    mean_x = cov_x*bsum;
    % Keep posteriors for the training set fixed though
    [mean_x, cov_x] = rollback_x(mean_x, cov_x, model);
    
    %
    % Compute posterior fox C
    %
    for i = 1:V
        if(missing_views(i))
            continue;
        end
        [Gamma{i}, H{i}]  = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y{i}, gamma_new{i}, Ltilde{i});
        cov_c{i} = inv(Gamma{i} + epsilon*J*J' + invOmega);
        mean_c{i} = gamma_new{i}*H{i}*cov_c{i}';      
    end
    % Keep posteriors for the training set fixed though
    [mean_c, cov_c] = rollback_c(mean_c, cov_c, model);
    
    if(debug_on)
        %For debug !        
        msg = sprintf('E- cycle: %d. y estimate: %f, %f\n', cycles, ...
            (mean_y{1})*model.norm_divide{1} + model.norm_minus{1}, ...
            (mean_y{2})*model.norm_divide{2} + model.norm_minus{2});
        fprintf(msg);
        tt = reshape(mean_x, [2, model.N_train+1]);
        col = ones(1, model.N_train+1);
        col(end) = 2;
        col(logical(nn)) = 3;
        scatter(tt(1, :), tt(2, :), 20, col);
        title(msg);
        getframe();
    end
end
mean_x_output = mean_x((end-dx+1):end);
cov_x_output = cov_x((end-dx+1):end, (end-dx+1):end);
% Normalize the y estimates (in case there where any)
mean_y_output = cell(1, V);
cov_y_output = cell(1, V);
for i = 1:V
    if(~missing_views(i))
        continue;
    end
    mean_y_output{i} = mean_y{i}*model.norm_divide{i} + model.norm_minus{i};
    cov_y_output{i} = cov_y{i} * model.norm_divide{i}^2;
    if(debug_on && isscalar(mean_y_output{i}))
        fprintf('Est for y (view #%d#): %.2f +- %.2f\n', i, mean_y_output{i}, sqrt(cov_y_output{i}));
    end
end
%
% Done. The rest is for debug
%
% Print x for neighbours as a check (works for dx=2)
if(debug_on)
    mean_x_new = mean_x_output;
    fprintf('#%d# Own x: %.3f %.3f\n', 0, mean_x_new(1), mean_x_new(2));
    valid_neighbours = find(nn);
    for j = valid_neighbours
        n_st = (j-1)*dx + 1;
        fprintf('\tneighbour n.%d: %.3f %.3f\n', j, mean_x(n_st), mean_x(n_st+1));
    end
end
return;

function [mean_x, cov_x] = rollback_x(mean_x, cov_x, model)
trainsize = size(model.cov_x, 1);
cov_x(1:trainsize, 1:trainsize) = model.cov_x;
trainsize = numel(model.mean_x);
mean_x(1:trainsize) = model.mean_x;
return;

function [mean_c, cov_c] = rollback_c(mean_c, cov_c, model)
V = numel(cov_c);
for i = 1:V
    trainsize = size(model.cov_c{i}, 1);
    cov_c{i}(1:trainsize, 1:trainsize) = model.cov_c{i};
end
for i = 1:V
    trainsize = size(model.mean_c{i}, 2);
    mean_c{i}(:, 1:trainsize) = model.mean_c{i};
end
return;

function [mean_c cov_c] = meanparams_c(model, neighbours)
V = numel(model.cov_c);
mean_c = cell(1, V);
cov_c = cell(1, V);
dx = model.dx;
neighbour_ids = find(neighbours);
num_nn = numel(neighbour_ids);
for i = 1:V
    cov_c{i} = zeros(dx, dx);
    mean_c{i} = zeros(model.dy{i}, dx);
    for n = neighbour_ids
        n_start = (n-1)*dx + 1;
        n_end = n_start + dx - 1;
        cov_c{i} = cov_c{i} + model.cov_c{i}(n_start:n_end, n_start:n_end);
        mean_c{i} = mean_c{i} + model.mean_c{i}(:, n_start:n_end);
    end
    cov_c{i} = cov_c{i} / num_nn;
    mean_c{i} = mean_c{i} / num_nn;
end
return;

function [mean_c, cov_c] = addnew_c(model, mean_c_init, cov_c_init)
V = numel(model.mean_c);
mean_c = model.mean_c;
cov_c = model.cov_c;
dx = model.dx;
for i = 1:V
    cov_c{i}((end+1):(end+dx), (end+1):(end+dx)) = cov_c_init{i};
    mean_c{i}(:, (end+1):(end+dx)) = mean_c_init{i};
end
return;

function [mean_x cov_x] = meanparams_x(model, neighbours)
dx = model.dx;
neighbour_ids = find(neighbours);
num_nn = numel(neighbour_ids);
%
cov_x = zeros(dx, dx);
mean_x = zeros(dx, 1);
for n = neighbour_ids
    n_start = (n-1)*dx + 1;
    n_end = n_start + dx - 1;
    cov_x = cov_x + model.cov_x(n_start:n_end, n_start:n_end);
    mean_x = mean_x + model.mean_x(n_start:n_end);
end
cov_x = cov_x / num_nn;
mean_x = mean_x / num_nn;
return;

function [mean_x, cov_x] = addnew_x(model, mean_x_init, cov_x_init)
mean_x = model.mean_x;
cov_x = model.cov_x;
dx = model.dx;
cov_x((end+1):(end+dx), (end+1):(end+dx)) = cov_x_init;
mean_x((end+1):(end+dx)) = mean_x_init;
return;

% function res = meanparams(modelvalues, neighbours, datum_size)
% neighbours = logical(neighbours);
% assert(mod(numel(modelvalues), datum_size) == 0);
% N = floor(numel(modelvalues) / datum_size);
% modelvalues_reshaped = reshape(modelvalues, [datum_size, N]);
% neighbourvalues = modelvalues_reshaped(:, neighbours);
% assert(size(neighbourvalues, 1) == datum_size);
% assert(size(neighbourvalues, 2) == nnz(neighbours));
% %scatter(modelvalues_reshaped(1, :), modelvalues_reshaped(2, :))
% %hold on;
% %scatter(neighbourvalues(1, :), neighbourvalues(2, :), 'r')
% 
% % Take the mean of the neighbours
% res = mean(neighbourvalues, 2);
% return;

function A = argmin(X)
[~, A] = min(X);
return;

function res = filter_nns(model, mean_x_init, nn, kstart, kmin)
assert(kmin < kstart);
[~, I] = sort(abs(mean_x_init - model.mean_x));
nn_new = zeros(size(nn));
k = kstart;
kmin = min(kmin, nnz(nn));
while 1
    nn_new(I(1:k)) = 1;
    res = nn_new & nn;
    if(nnz(res) >= kmin)
        break;
    end
    k = k + 1;
end
return;
