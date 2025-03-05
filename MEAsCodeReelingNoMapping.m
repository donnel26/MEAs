delete(gcp('nocreate'));   % this line closes the parallel pool
%% 
clear; clc;
rng(1001); 
Ngene_df = readtable('NgeneSim.csv');

theta = 1;
alpha_1 = 0.2; 
alpha_2 = -.3; 
alpha_3 = -.1;
alpha_4 = 0.15;
alpha_5 = 0.5; 
gamma_0 = .5;
mu_0 = -0.05; 
phi_0 = 0.04*4; 
gamma_1 = -0.012*2;   
phi_1 = -0.002; 

choice = Ngene_df.ChoiceSituation; 
BL = Ngene_df.yes_bl;             
price = Ngene_df.yes_price_;     
BL_prob = Ngene_df.yes_bl_yes_prob; 
price_prob = Ngene_df.yes_price__yes_prob; 

% Simulate W data 
num_samples = height(Ngene_df);
num_W_vars = 5; % Number of W variables
unique_choices = unique(choice); 
num_choices = length(unique_choices); 
W_simulated = zeros(num_samples, num_W_vars);  
Z_simulated = zeros(num_samples, 1); 
group_W_values = rand(num_choices, num_W_vars);   
group_Z_values = randi([0, 30], num_choices, 1);  

for i = 1:num_choices  
    group_mask = (choice == unique_choices(i));  
    W_simulated(group_mask, :) = repmat(group_W_values(i, :), sum(group_mask), 1);  
    Z_simulated(group_mask) = group_Z_values(i);   
end  

q_simulated = (randi([0, 1], num_samples, 1) * 1000 + 1000)/1000; % Simulated q values (1000 or 2000)

% Calculate interactions
bZ_simulated = BL .* Z_simulated; % Interaction term between BL and Z
qZ_simulated = q_simulated .* Z_simulated; % Interaction term between q and Z

% Add simulated data to Ngene_df
for i = 1:num_W_vars
    Ngene_df.("W" + string(i)) = W_simulated(:, i); 
end

% Add simulated data to Ngene_df
for i = 1:num_W_vars
    Ngene_df.("W" + string(i)) = W_simulated(:, i); 
end
Ngene_df.Z_simulated = Z_simulated;          
Ngene_df.q_simulated = q_simulated;         
Ngene_df.bZ_simulated = bZ_simulated;        
Ngene_df.qZ_simulated = qZ_simulated;        
   
permit_dummy = findgroups(Ngene_df.q_simulated, Ngene_df.yes_bl, Ngene_df.yes_price_);
Ngene_df.permit_dummy = permit_dummy;

X = [W_simulated, BL, bZ_simulated, q_simulated,  qZ_simulated, price];
param_names = {'Intercept', 'W1', 'W2', 'W3', 'W4', 'W5', 'BL', 'bZ_simulated', 'q_simulated',  'qZ_simulated','Price'};

% Utility functions
%  CJR code through line 81
alpha = [alpha_1 alpha_2 alpha_3 alpha_4 alpha_5];
p = rand(num_samples,1);
e = log(p./(1 - p));
H = theta + BL.*(gamma_0 + gamma_1.*Z_simulated) + q_simulated.*(phi_0 + phi_1.*Z_simulated);
V = W_simulated*alpha' + H + mu_0*price + e;
D = zeros(num_samples, 1); 
D(V > 0) = 1;

share = splitapply(@mean, D, permit_dummy);
Ngene_df.share = share(permit_dummy);

% Estimate logit model
[logit_model, ~, stats] = glmfit(X, D, 'binomial', 'link', 'logit');
for i = 1:length(param_names)
    fprintf('%-18s %-12.4f %-12.4f\n', param_names{i}, logit_model(i), stats.se(i)); 
end

% Confirm that estimated model predicts observed choices exactly
pifit = glmval(logit_model,X,'logit','Size',num_samples);
sum(D)/num_samples
pihat = exp([ones(num_samples,1) X]*logit_model)./(1 + exp([ones(num_samples,1) X]*logit_model));
mean(pihat)
%%
% Extract covariance matrix and coefficients
covariance_matrix = stats.covb;
theta = logit_model(1);
coefficients = logit_model(2:end);
alpha_1 = coefficients(1); 
alpha_2 = coefficients(2); 
alpha_3 = coefficients(3); 
alpha_4 = coefficients(4); 
alpha_5 = coefficients(5); 
gamma_0 = coefficients(6); % bag
gamma_1  = coefficients(7);
phi_0 = coefficients(8); % quota
phi_1= coefficients(9); % bSz
mu = coefficients(10); % qz
num_pay_1 = sum(D == 1);
sigma = num_pay_1 / num_samples;
theta_hat = theta / sigma;

% Drop Duplicates 
[~, ia] = unique(Ngene_df.ChoiceSituation, 'stable'); %% Identifies the first occurence of unique ChoiceSituation
unique = Ngene_df(ia, :); %% Creates a new table, but just with unique ChoiceSituaitons

% Krinsky Robb Stuff
beta_hat = [theta; alpha_1; alpha_2; alpha_3; alpha_4; alpha_5; gamma_0; gamma_1; phi_0; phi_1; mu];
C = chol(covariance_matrix, 'lower');
N_sim = 5000; % number of simulations
K = length(beta_hat); 
xK = randn(K, N_sim);
beta_hat_sim = repmat(beta_hat, 1, N_sim);
beta_d = beta_hat_sim + C * xK;

b_values = [1, 2, 3];
Q_values = [1000, 2000]; 
N_j = num_samples;  

sigma_matrices = zeros(length(b_values), length(Q_values), N_sim);
pi_ij_matrices = zeros(N_j, length(b_values), length(Q_values), N_sim);
%% 
% For each draw of the betas, calculate the choice probability
thetahat = beta_d(1,:);
alphahat = beta_d(2:6,:);
gamma_0hat = beta_d(7,:);
gamma_1hat = beta_d(8,:);
phi_0hat = beta_d(9,:);
phi_1hat = beta_d(10,:);
muhat = beta_d(11,:);
%% 

% I just chose an arbitary b, q, l; you'd have to repeat this for each
% combination
BLsim = 2;
qsim = 1;  % Divide by 1000
pricesim = 20;

Hhat = thetahat.*sigma + BLsim.*(gamma_0hat + gamma_1hat.*Z_simulated) + ...
    qsim.*(phi_0hat + phi_1hat.*Z_simulated);
Vsim = W_simulated*alphahat + Hhat + muhat*pricesim;
pihat = 1./(1 + exp(-Vsim));

dsigmadb = (1/N_j.*sum((gamma_0hat + gamma_1hat.*Z_simulated).*pihat.*(1 - pihat)))./...
    (1 + 1/N_j.*sum(thetahat.*pihat.*(1 - pihat)));
dsigmadq = (1/N_j*sum((phi_0hat + phi_1hat.*Z_simulated).*pihat.*(1 - pihat)))./...
    (1 + 1/N_j.*sum(thetahat.*pihat.*(1 - pihat)));
dWTPdb = mean(1./abs(muhat).*(gamma_0hat + gamma_1hat.*Z_simulated + thetahat.*dsigmadb));
dWTPdq = mean(1./abs(muhat).*(phi_0hat + phi_1hat.*Z_simulated + thetahat.*dsigmadq));
[prctile(dWTPdb,2.5) mean(dWTPdb) prctile(dWTPdb,97.5)]

%%
% Initialize Parallel Pool
parpool(4);
tic
sigma_matrices = zeros(length(b_values), length(Q_values), N_sim);
dsigma_dq_values = zeros(length(b_values), length(Q_values) - 1, N_sim);
dS_dq_values = zeros(N_j, length(b_values), length(Q_values) - 1, N_sim);
update_interval = 10; % Print update every 10 iterations
parfor s = 1:N_sim
   iter_tic = tic; % Start timing the iteration
 task = getCurrentTask(); % Get worker info
 workerID = task.ID; % Worker ID (only works inside parfor)
    % PARAMS %
    alpha_0_sim = beta_d(1, s);
    alpha_1_sim = beta_d(2, s);
    alpha_2_sim = beta_d(3, s);
    alpha_3_sim = beta_d(4, s);
    alpha_4_sim = beta_d(5, s);
    alpha_5_sim = beta_d(6, s);
    gamma_0_sim = beta_d(7, s);
    mu_sim      = beta_d(8, s);
    phi_0_sim   = beta_d(9, s);
    gamma_1_sim = beta_d(10, s);
    phi_1_sim   = beta_d(11, s);

        % MATRICES %
    local_sigma_matrix = zeros(length(b_values), length(Q_values));
    local_dsigma_dq = zeros(length(b_values), length(Q_values) - 1);
    local_dS_dq = zeros(N_j, length(b_values), length(Q_values) - 1);
    local_dsigma_db = zeros(length(b_values)-1, length(Q_values));
    local_dS_db = zeros(N_j, length(b_values)-1, length(Q_values));
  
    % SIGMAS %
    for i = 1:length(b_values)
        for j = 1:length(Q_values)
            BL_val = b_values(i);
            q_val  = Q_values(j);
            diff = 1;
            sigma_local = 0.1;  % initial guess
            k = 0.6;
            iter = 0;
            max_iter = 100;
            while diff >= 1e-6 && iter < max_iter
                pi_ij = 1 ./ (1 + exp( alpha_1_sim * W_simulated(:,1) + alpha_2_sim * W_simulated(:,2) + ...
                           alpha_3_sim * W_simulated(:,3) + alpha_4_sim * W_simulated(:,4) + ...
                           alpha_5_sim * W_simulated(:,5)) + theta_hat * sigma_local + ...
                           (gamma_0_sim + gamma_1_sim * Z_simulated) * BL_val + ...
                           (phi_0_sim + phi_1_sim * Z_simulated) * q_val - mu_sim * price);
                sigma_new = sum(pi_ij)/N_j * (1 - k) + k * sigma_local;
                diff = max(abs(sigma_local - sigma_new));
                sigma_local = sigma_new;
                iter = iter + 1;
                  % Print update every `update_interval` iterations
                if mod(iter, update_interval) == 0 || iter == 1
                    fprintf('Worker %d: Simulation %d, (i=%d, j=%d) - Iteration %d/%d\n', ...
                        workerID, s, i, j, iter, max_iter);
                end
            end
           local_sigma_matrix(i, j) = sigma_local;
        end
    end
   
% dsigma/dq % 
    for i = 1:length(b_values)
        for j = 1:(length(Q_values)-1)
            local_dsigma_dq(i, j) = (local_sigma_matrix(i, j + 1) - local_sigma_matrix(i, j)) /(Q_values(j + 1) - Q_values(j));
        end
    end

   
% dsigma/db % 
    for j = 1:length(Q_values)  % For each Q value
        for i = 1:(length(b_values) - 1)  % For each change in b
             local_dsigma_db(i, j) = (local_sigma_matrix(i + 1, j) - local_sigma_matrix(i, j)) / (b_values(i + 1) - b_values(i));
        end
    end


% ds/dq % 
for i = 1:length(b_values)
        for j = 1:(length(Q_values) - 1)
            local_dS_dq(:, i, j) = (1 / mu_sim) * ((phi_0_sim + phi_1_sim * Z_simulated) + theta_hat * local_dsigma_dq(i, j));
        end
end

% ds/db % 
    for j = 1:length(Q_values)
        for i = 1:(length(b_values) - 1)
            local_dS_db(:, i, j) = (1 / mu_sim) * ((gamma_0_sim + gamma_1_sim * Z_simulated) + theta_hat * local_dsigma_db(i, j));
        end
    end

 % Store results %
    sigma_matrices(:,:,s) = local_sigma_matrix;
    dsigma_dq_values(:,:,s) = local_dsigma_dq;
    dS_dq_values(:,:,:,s) = local_dS_dq;
     dsigma_db_values(:,:,s) = local_dsigma_db;
    dS_db_values(:,:,:,s) = local_dS_db;
% Print progress update
    fprintf('Worker %d: Simulation %d/%d completed in %.2f seconds.\n', workerID, s, N_sim, toc(iter_tic));
end

toc
delete(gcp('nocreate'));
%% CONFIDENCE INTERVALS & SUMMARY STATISTICS USING KIRINSKY & ROBB
sorted_dS_dq_values = sort(dS_dq_values, 4); % Sort along simulations (4th dim)
CI_dS_dq = zeros(2, length(Q_values)-1);
stats_dS_dq = zeros(3, length(Q_values)-1);

sorted_dS_db_values = sort(dS_db_values, 4); % idk if rightf
CI_dS_db = zeros(2, length(b_values)-1);
stats_dS_db = zeros(3, length(b_values)-1);


for j = 1:(length(Q_values)-1)
    vals = sorted_dS_dq_values(:, :, j, :); % Extract across simulations
    vals = vals(:); 
    % Compute Confidence Intervals and Summary Stats
    CI_dS_dq(:, j) = prctile(vals, [2.5, 97.5]); % 95% CI
    stats_dS_dq(1, j) = mean(vals);   % Mean
    stats_dS_dq(2, j) = prctile(vals, 25); % 25th percentile
    stats_dS_dq(3, j) = prctile(vals, 75); % 75th percentile
end

for i = 1:(length(b_values)-1)
    vals = sorted_dS_db_values(:, i, :, :);
    vals = vals(:); 
    % Compute Confidence Intervals and Summary Stats
    CI_dS_db(i,:) = prctile(vals, [2.5, 97.5]); % 95% CI
    stats_dS_db(1, i, :) = mean(vals);   % Mean
    stats_dS_db(2, i, :)  = prctile(vals, 25); % 25th percentile
    stats_dS_db(3, i, :)  = prctile(vals, 75); % 75th percentile
end


disp('95% Confidence Intervals for dS/dq:');
disp(CI_dS_dq);

disp('Summary Statistics for dS/dq:');
disp('Row 1: Mean, Row 2: 25th Percentile, Row 3: 75th Percentile');
disp(stats_dS_dq);


disp('95% Confidence Intervals for dS/db:');
disp(CI_dS_db);

disp('Summary Statistics for dS/db:');
disp('Row 1: Mean, Row 2: 25th Percentile, Row 3: 75th Percentile');
disp(stats_dS_db);

disp('Logistic Regression Coefficients:');
disp(logit_model);
disp(['Constant (intercept): ', num2str(constant)]);
disp(['Share of pay=1 (sigma): ', num2str(sigma)]);
disp(['Theta_hat (θhat_0): ', num2str(theta_hat)]);





%% 




















% Calculate averages over simulations
average_sigma = mean(sigma_matrices, 3); 
average_pi_ij = mean(pi_ij_matrices, 4); 
% List all relevant variables you want to save
variables_to_save = { ...
    'BL', 'bZ_simulated', 'num_pay_1', ...
    'BL_prob', 'b_0', 'num_samples', ...
    'C', 'b_BL', 'param_names', ...
    'D', 'b_price', 'permit_dummy', ...
    'E0', 'b_values', 'phi_0', ...
    'E1', 'beta_d', 'phi_1', ...
    'K', 'beta_hat', 'pi_ij_matrices', ...
    'N_j', 'choice', 'price', ...
    'N_sim', 'coefficients', 'price_prob', ...
    'Ngene_df', 'cons', 'prob', ...
    'Q_values', 'constant', 'qZ_simulated', ...
    'U_no', 'covariance_matrix', 'q_simulated', ...
    'U_yes', 'data', ...
    'W_simulated', 'deviance', 'share', ...
    'X', 'gamma_0', 'sigma', ...
    'Z_simulated', 'gamma_1', 'sigma_matrices', ...
    'alpha_0', 'group_W_values', 'stats', ...
    'alpha_1', 'group_Z_values', 'theta_hat', ...
    'alpha_2', 'group_mask', 'total_obs', ...
    'alpha_3', 'i', 'unique', ...
    'alpha_4', 'ia', 'unique_choices', ...
    'alpha_5', 'logit_model', 'variable_matrix', ...
    'mu', 'xK', ...
    'average_pi_ij', 'num_W_vars', ...
    'average_sigma', 'num_choices'};

% Save all relevant data in one file
save('MEAsPart1data.mat', variables_to_save{:}); % Save all variables to this file
