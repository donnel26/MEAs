
delete(gcp('nocreate'));   % this line closes the parallel pool
clear; clc;
rng(1001); 
Ngene_df = readtable('C:\\Users\\donnel26\\OneDrive - purdue.edu\\RA assignment FA ''23\\SHC\\NgeneSimMEAs.csv');
alpha_0 = 0; 
alpha_1 = 0.2; 
alpha_2 = 0.3; 
alpha_3 = 0.1;
alpha_4 = 0.15;
alpha_5 = 0.05; 
gamma_0 = -0.5;    
b_price = -0.05; 
phi_0 = 0.4; 
gamma_1 = 0.6;   
phi_1 = 0.2; 
param_names = {'Intercept', 'W1', 'W2', 'W3', 'W4', 'W5', 'BL', 'Price', 'q_simulated', 'bZ_simulated', 'qZ_simulated'};

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

% Simulate q data 
q_simulated = randi([0, 1], num_samples, 1) * 10000;


% Calculate interactions
bZ_simulated = BL .* Z_simulated; % BL * Z
qZ_simulated = q_simulated .* Z_simulated; % q * Z

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

X = [W_simulated, BL, price, q_simulated, bZ_simulated, qZ_simulated];
E1 = -log(-log(rand(num_samples, 1)));
E0 = -log(-log(rand(num_samples, 1))); 

% Utility functions
U_yes = (alpha_0 + alpha_1 * W_simulated(:,1) + alpha_2 * W_simulated(:,2) +  alpha_3 * W_simulated(:,3) + alpha_4 * W_simulated(:,4) + alpha_5 * W_simulated(:,5)) + ...
        (gamma_1 * Ngene_df.yes_bl_yes_prob + b_price * price_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E1;
U_no =  (alpha_0 + alpha_1 * W_simulated(:,1) + alpha_2 * W_simulated(:,2) +  alpha_3 * W_simulated(:,3) + alpha_4 * W_simulated(:,4) + alpha_5 * W_simulated(:,5)) + ...
        (gamma_1 * Ngene_df.yes_bl_yes_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E0;
D = zeros(num_samples, 1);
D(U_yes > U_no) = 1;

share = splitapply(@mean, D, permit_dummy);
Ngene_df.share = share(permit_dummy);

% Estimate logit model
[logit_model, ~, stats] = glmfit(X, D, 'binomial', 'link', 'logit');
for i = 1:length(param_names)
    fprintf('%-18s %-12.4f %-12.4f\n', param_names{i}, logit_model(i), stats.se(i)); 
end

% Extract covariance matrix and coefficients
covariance_matrix = stats.covb;
alpha_0 = logit_model(1);
coefficients = logit_model(2:end);
alpha_1 = coefficients(1); 
alpha_2 = coefficients(2); 
alpha_3 = coefficients(3); 
alpha_4 = coefficients(4); 
alpha_5 = coefficients(5); 
gamma_0 = coefficients(6); % bag
mu = coefficients(7);
phi_0 = coefficients(8); % quota
gamma_1 = coefficients(9); % bz
phi_1 = coefficients(10); % qz
constant = alpha_0;
num_pay_1 = sum(D == 1);
sigma = num_pay_1 / num_samples;
theta_hat = constant / sigma;

beta_hat = [alpha_0; alpha_1; alpha_2; alpha_3; alpha_4; alpha_5; gamma_0; mu; phi_0; gamma_1; phi_1];
C = chol(covariance_matrix, 'lower');
N_sim = 5000; % number of simulations
K = length(beta_hat); 
xK = randn(K, N_sim);
beta_hat_sim = repmat(beta_hat, 1, N_sim);
beta_d = beta_hat_sim + C * xK;

b_values = [1, 2, 3];

Q_values = [0, 10000];
N_j = num_samples; 

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

% ds/dq % 
for i = 1:length(b_values)
        for j = 1:(length(Q_values) - 1)
            local_dS_dq(:, i, j) = (1 / mu_sim) * ((phi_0_sim + phi_1_sim * Z_simulated) + theta_hat * local_dsigma_dq(i, j));
        end
end

 % Store results %
    sigma_matrices(:,:,s) = local_sigma_matrix;
    dsigma_dq_values(:,:,s) = local_dsigma_dq;
    dS_dq_values(:,:,:,s) = local_dS_dq;
% Print progress update
    fprintf('Worker %d: Simulation %d/%d completed in %.2f seconds.\n', workerID, s, N_sim, toc(iter_tic));
end

toc
delete(gcp('nocreate'));
%% CONFIDENCE INTERVALS & SUMMARY STATISTICS USING KIRINSKY & ROBB
sorted_dS_dq_values = sort(dS_dq_values, 4); % Sort along simulations (4th dim)
CI_dS_dq = zeros(2, length(Q_values)-1);
stats_dS_dq = zeros(3, length(Q_values)-1);

for j = 1:(length(Q_values)-1)
    vals = sorted_dS_dq_values(:, :, j, :); % Extract across simulations
    vals = vals(:); 
    % Compute Confidence Intervals and Summary Stats
    CI_dS_dq(:, j) = prctile(vals, [2.5, 97.5]); % 95% CI
    stats_dS_dq(1, j) = mean(vals);   % Mean
    stats_dS_dq(2, j) = prctile(vals, 25); % 25th percentile
    stats_dS_dq(3, j) = prctile(vals, 75); % 75th percentile
end

%% DISPLAY RESULTS
disp('95% Confidence Intervals for dS/dq:');
disp(CI_dS_dq);

disp('Summary Statistics for dS/dq:');
disp('Row 1: Mean, Row 2: 25th Percentile, Row 3: 75th Percentile');
disp(stats_dS_dq);

disp('Logistic Regression Coefficients:');
disp(logit_model);
disp(['Constant (intercept): ', num2str(constant)]);
disp(['Share of pay=1 (sigma): ', num2str(sigma)]);
disp(['Theta_hat (θhat_0): ', num2str(theta_hat)]);