% compute Ltilde by a chosen option
% (1) its definition
% (2) singular value decomposition of L
% mijung wrote on the 19th of Oct, 2015

function [result] = compute_Ltilde(L, epsilon, gamma, opt_dec)

% decompose Ltilde using singular value decomposition of L
% definition of  Ltilde = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);
%  we can re-write Ltilde = Ltilde_epsilon + 1/(2*gamma)* Ltilde_L

n = size(L,1);
V = numel(gamma);

if opt_dec==0
    for i = 1:V
      result{i} = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma(i)*L);
    end
else
    
    [U_L, D_L, V_L] = svd(L); % L := U_L*D_L*V_L'
    
    Depsilon_inv = zeros(n,n);    
    sign_sin_val = V_L(:,end)'*U_L(:,end); % because matlab sometimes flips the sign of singular vectors
    Depsilon_inv(n,n) = sign_sin_val *1./(epsilon*n); 
    Ltilde_epsilon = V_L*Depsilon_inv*U_L' ;
    
    D_L_inv = zeros(n,n);
    D_L_inv(1:n-1, 1:n-1) = diag(1./diag(D_L(1:n-1, 1:n-1)));
    Ltilde_L = V_L*D_L_inv*U_L';
    
    % check if they match
    for i = 1:V
      result{i} = struct();
      result{i}.Ltilde = Ltilde_epsilon + 1./(2*gamma{i})* Ltilde_L;
    end
    % Copying v times is redundant but helps simplifying (a bit) the code later
    for i = 1:V
      result{i}.Ltilde_epsilon = Ltilde_epsilon;
      result{i}.Ltilde_L = Ltilde_L;
      result{i}.eigL = diag(D_L); 
    end
end

