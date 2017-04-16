function [results, op ] = lllvm_1ep(Y, op, visu)
% Multiview LL-LVM . Extended from code for LL-LVM model (https://github.com/mijungi/lllvm)
%
%   - Assume V^-1 = gamma*I for all views (predicision matrix in the likelihood).
%Input:
% - Y: A cell of dy_v x n matrices of observed views.
% - op: a struct containing various options. See the code with lines containing
%   myProcessOptions or isfield for possible options.
%

% random seed. Default to 1.
seed = myProcessOptions(op, 'seed', 4);
fprintf('Seed=%d\n', seed);
oldRng = rng();
rng(seed);

% check if input is a single view
if(~iscell(Y))
    Ys = cell(1, 1);
    Ys{1} = Y;
    Y = Ys;
    clear Ys;
end

% Number of input views
V = numel(Y);
for i = 1:V
  [dy{i}, n] = size(Y{i});
  % Using original normalization, and saving normalization params.
  op.norm_minus{i} = mean(Y{i}, 2);
  Y{i} = bsxfun(@minus, Y{i}, op.norm_minus{i});
  op.norm_divide{i} = max(abs(Y{i}(:)));  
  Y{i} = Y{i}/op.norm_divide{i};  
end

% maximum number of EM iterations to perform.
max_em_iter = myProcessOptions(op, 'max_em_iter', 200);
abs_tol = myProcessOptions(op, 'abs_tol', 1e-4);
if ~isfield(op, 'G')
    error('option G for the neighbourhood graph is mandatory');
end
G = op.G;
assert(all(size(G) == [n,n]), 'Size of G must be n x n.');
% dimensionality of the reduced representation X. This must be less than dy.
% This option is mandatory.
if ~isfield(op, 'dx')
    error('option dx (reduced dimensionality) is mandatory.');
end
dx = op.dx;
assert(dx > 0);

L = diag(sum(G, 1)) - G;
invOmega = kron(2*L, eye(dx));
% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
alpha0 = myProcessOptions(op, 'alpha0', 1);
assert(alpha0 > 0, 'require alpha0 > 0');


% invPi is Pi^-1 where p(x|G, alpha) = N(x | 0, Pi) (prior of X).
invPi = alpha0*eye(n*dx) + invOmega;

% Initial value of gamma.
for i = 1:V
  gamma0{i} = myProcessOptions(op, 'gamma0', 1);
  assert(gamma0{i} > 0, 'require gamma0 > 0');
end
% epsilon is a positive real number in the likelihood term specifying the amount 
% to penalize deviation of the mean of the data from 0. 
epsilon = myProcessOptions(op, 'epsilon', 1e-4);


% A recorder is a function handle taking in a struct containing all
% intermediate variables in LL-LVM in each EM iteration and does something.
% lllvm will call the recorder at the end of every EM iteration with a struct
% containing all variables if one is supplied.
%
% For example, a recorder may just print all the variables in every iteration.
recorder = myProcessOptions(op, 'recorder', []);

J = kron(ones(n,1), eye(dx));
for i = 1:V
  % cov_c0 hyperparameter 
  cov_c0{i} = myProcessOptions(op, 'cov_c0', inv(epsilon*(J*J') + invOmega) );
  % mean_c0 hyperparameter
  mean_c0{i} = myProcessOptions(op, 'mean_c0', randn(dy{i}, dx*n)*cov_c0{i}');
end
disp(mean_c0{1}(1));
% collect all options used.
op.seed = seed;
op.max_em_iter = max_em_iter;
op.alpha0 = alpha0;
op.gamma0 = gamma0{1};
op.epsilon = epsilon;
% We will not collect the recorder as it will make a saved file big.


%========= Start EM
gamma_new = gamma0;
invPi_new = invPi;


opt_dec = 1; % using decomposition
[Ltilde] = compute_Ltilde(L, epsilon, gamma_new, opt_dec);
eigv_L = Ltilde{1}.eigL;

cov_c = cov_c0;
mean_c = mean_c0;

% to store results
for i = 1:V
  meanCmat{i} = zeros(n*dx*dy{i}, max_em_iter);
  covCmat{i} = zeros(n*dx, max_em_iter);
  gammamat{i}  = zeros(max_em_iter,1);  
end
meanXmat = zeros(n*dx, max_em_iter);
covXmat = zeros(n*dx, max_em_iter);
alphamat = zeros(max_em_iter,1);

% vars to store lower bound
prev_lwb = inf();
% lower bound in every iteration
lwbs = [];
for i_em = 1:max_em_iter
    
    tic;
    %% (1) E step
    invcov_x = invPi_new;
    bsum = 0;
    for i = 1:V
      [A{i}, b{i}] = compute_suffstat_A_b(G, mean_c{i}, cov_c{i}, Y{i}, gamma_new{i}, epsilon);
      invcov_x = invcov_x + A{i};
      bsum = bsum + b{i};
    end
    cov_x = inv(invcov_x);
    mean_x = cov_x*bsum;

    for i = 1:V
      [Gamma{i}, H{i}, Gamma_L{i}]  = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y{i}, gamma_new{i}, Ltilde{i});
      cov_c{i} = inv(Gamma{i} + epsilon*J*J' + invOmega);
      mean_c{i} = gamma_new{i}*H{i}*cov_c{i}';      
    end
    %% (2) M step
    lwb_likelihood = 0; %have to check 
    lwb_C = 0;
    for i = 1:V
      [acc_lwb_likelihood, gamma_new{i}] = exp_log_likeli_update_gamma(mean_c{i}, cov_c{i}, H{i}, Y{i}, L, epsilon, Ltilde{i}, Gamma_L{i});
      acc_lwb_C = negDkl_C(mean_c{i}, cov_c{i}, invOmega, J, epsilon);
      lwb_likelihood = lwb_likelihood + acc_lwb_likelihood;
      lwb_C = lwb_C + acc_lwb_C;
    end
    [lwb_x, alpha_new] = negDkl_x_update_alpha(mean_x, cov_x, invOmega, eigv_L);
    
    % (2.half) update invPi using the new alpha
    invPi_new = alpha_new*eye(n*dx) + invOmega;
    
    %% (3) compute the lower bound
    lwb = lwb_likelihood + lwb_C + lwb_x;

    iter_time = toc();
    
    display(sprintf('EM iteration %d/%d. Took: %.3f s. Lower bound: %.3f', ...
        i_em, max_em_iter, iter_time, lwb));
    
    lwbs(end+1) = lwb;
    
    if(exist('visu', 'var') && visu)
        scatter(mean_x, Y{2}(2, :));
        axis([-1 1 -1 1]);        
        title(sprintf('x vs true intrinsic param.(EM iter %d)', i_em));
        pause(.01);
    end
    
    % call the recorder
    if ~isempty(recorder)
        % collect all the variables into a struct to pass to the recorder
        % take everything from the op except the recorder itself
        op2 = rmfield(op, 'recorder');
        state = op2;
        state.Y = Y;
        state.i_em = i_em;
        state.alpha = alpha_new;
        state.gamma = gamma_new;
        state.Gamma = Gamma;
        state.H = H;
        state.cov_c = cov_c;
        state.mean_c = mean_c;
        state.cov_x = cov_x;
        state.mean_x = mean_x;
        state.A = A;
        state.b = b;
        state.lwb = lwb;
        
        % call the recorder
        recorder(state);
    end
    
    % check increment of the lower bound.
    if i_em >= 2 && abs(lwb - prev_lwb) < abs_tol
        % stop if the increment is too low.
        break;
    end
    prev_lwb = lwb;
    
end %end main EM loop


% construct the results struct
for i = 1:V
    r.cov_c{i} = cov_c{i};
    r.mean_c{i} = mean_c{i};
    r.gamma{i} = gamma_new{i};
end
r.alpha = alpha_new;
r.cov_x = cov_x;
r.mean_x = mean_x;
r.G = G;
r.dx = dx;

r.lwbs = lwbs(1:i_em);
r.epsilon = epsilon;
% Save normalization params
for i = 1:V
    r.norm_minus{i} = op.norm_minus{i};
    r.norm_divide{i} = op.norm_divide{i};
end
results = r;

rng(oldRng);
end

