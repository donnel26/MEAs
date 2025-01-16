delete(gcp('nocreate'));   % this line closes the parallel pool
%% 
clear; clc;
rng(1001); 
Ngene_df = readtable('C:\\Users\\Emma\\OneDrive - purdue.edu\\Desktop\\RA Fall 2023\\SHC\\NgeneSim.csv');

% Parameters
b_0 = -0.5;     %% change? 
b_BL = 0.6;   
b_price = -0.05; 
alpha_0 = 0; 
alpha_1 = 0.2; 
alpha_2 = 0.3; 
alpha_3 = 0.1;
alpha_4 = 0.15;
alpha_5 = 0.05; 
phi_0 = 0.4; 
phi_1 = 0.2; 
choice = Ngene_df.ChoiceSituation; 
BL = Ngene_df.yes_bl;             
prob = Ngene_df.yes_prob;        
price = Ngene_df.yes_price_;     
BL_prob = Ngene_df.yes_bl_yes_prob; 
price_prob = Ngene_df.yes_price__yes_prob; 

% Simulating W data 
num_samples = height(Ngene_df);
num_W_vars = 5; % Number of W variables to simulate

unique_choices = unique(choice);  %% new 
num_choices = length(unique_choices);  %% new 

W_simulated = zeros(num_samples, num_W_vars);  %% new 
Z_simulated = zeros(num_samples, 1);  %% new 

group_W_values = rand(num_choices, num_W_vars);   %% new 
group_Z_values = randi([0, 30], num_choices, 1);  %% new 

for i = 1:num_choices  %% new 
    group_mask = (choice == unique_choices(i));  %% new 
    W_simulated(group_mask, :) = repmat(group_W_values(i, :), sum(group_mask), 1);  %% new 
    Z_simulated(group_mask) = group_Z_values(i);  %% new 
end  %% new 

q_simulated = randi([0, 1], num_samples, 1) * 1000 + 1000; % Simulated q values (1000 or 2000)

% Calculate interactions
bZ_simulated = BL .* Z_simulated; % Interaction term between BL and Z
qZ_simulated = q_simulated .* Z_simulated; % Interaction term between q and Z

% Add simulated data to Ngene_df
for i = 1:num_W_vars
    Ngene_df.("W" + string(i)) = W_simulated(:, i); 
end

% Add the other simulated variables to the table
Ngene_df.Z_simulated = Z_simulated;          
Ngene_df.q_simulated = q_simulated;         
Ngene_df.bZ_simulated = bZ_simulated;        
Ngene_df.qZ_simulated = qZ_simulated;       
   
permit_dummy = findgroups(Ngene_df.q_simulated, Ngene_df.yes_bl, Ngene_df.yes_price_); % Grouping based on quota, bag, price
data.permit_dummy = permit_dummy;
Ngene_df.permit_dummy = permit_dummy;

X = [W_simulated, BL, price, q_simulated, bZ_simulated, qZ_simulated];
E1 = -log(-log(rand(height(Ngene_df), 1)));
E0 = -log(-log(rand(height(Ngene_df), 1))); 

% Utility functions
%U_yes = zeros(num_samples, 1); 
U_yes = (alpha_0 + alpha_1 * W_simulated(:, 1) + alpha_2 * W_simulated(:, 2) +  alpha_3 * W_simulated(:, 3) + alpha_4 * W_simulated(:, 4) + alpha_5 * W_simulated(:, 5)) + (b_BL * BL_prob + b_price * price_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E1;
U_no =   (alpha_0 + alpha_1 * W_simulated(:, 1) + alpha_2 * W_simulated(:, 2) +  alpha_3 * W_simulated(:, 3) + alpha_4 * W_simulated(:, 4) + alpha_5 * W_simulated(:, 5)) + (b_BL * BL_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E0;
% Decision variable
D = zeros(height(Ngene_df), 1);  % Initialize D
D(U_yes > U_no) = 1; % Assign 1 if U_yes > U_no
cons = zeros(height(Ngene_df), 1);

share = splitapply(@mean, D, permit_dummy);
% Add the share column back to the dataset
Ngene_df.share = share(permit_dummy);

% estimate logit model
logit_model = glmfit(X, D, 'binomial', 'link', 'logit');

disp('Logistic Regression Coefficients:');
disp(logit_model);
% Define parameter names for display
param_names = {'Intercept', 'W1', 'W2', 'W3', 'W4', 'W5', 'BL', 'Price', 'q_simulated', 'bZ_simulated', 'qZ_simulated'}; 

% Display coefficients with parameter names
for i = 1:length(logit_model)
    fprintf('%s: %.4f\n', param_names{i}, logit_model(i));
end

% Extract covariance matrix
[~, ~, stats] = glmfit(X, D, 'binomial', 'link', 'logit');
covariance_matrix = stats.covb;

%save coefficients
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
gamma_1 = coefficients(9); %bz
phi_1 = coefficients(10); %qz
constant = alpha_0;
num_pay_1 = sum(D == 1);
total_obs = height(Ngene_df); 
sigma = num_pay_1 / total_obs; %% sigma is share that pay
disp(['Constant (intercept): ', num2str(constant)]);
disp(['Share of pay=1: ', num2str(sigma)]);
theta_hat = constant / sigma;
disp(['θhat_0: ', num2str(theta_hat)]);

% Krinsky Robb Stuff
beta_hat = [alpha_0; alpha_1; alpha_2; alpha_3; alpha_4; alpha_5; gamma_0; mu; phi_0; gamma_1; phi_1];
variable_matrix = [W_simulated, BL, price, q_simulated, bZ_simulated, qZ_simulated];
C = chol(covariance_matrix, 'lower');
N_sim = 5000; % number of simulations
K = length(beta_hat); % number of params
xK = randn(K, N_sim);
beta_hat = repmat(beta_hat, 1, N_sim);
beta_d = beta_hat + C * xK;
b_values = [1, 2, 3];
Q_values = [1000, 2000]; 
N_j = height(Ngene_df); 
% Initialize sigma & pi_ij matrices for calculations
sigma_matrices = zeros(length(b_values), length(Q_values), N_sim);   
pi_ij_matrices = zeros(N_j, length(b_values), length(Q_values), N_sim);

%% 
parpool(4);

parfor s = 1:N_sim
    beta_hat = [alpha_0; alpha_1; alpha_2; alpha_3; alpha_4; alpha_5; gamma_0; mu; phi_0; gamma_1; phi_1];

    alpha_0_sim = beta_d(1, s);
    alpha_1_sim = beta_d(2, s);
    alpha_2_sim = beta_d(3, s);
    alpha_3_sim = beta_d(4, s);
    alpha_4_sim = beta_d(5, s);
    alpha_5_sim = beta_d(6, s);
    gamma_0_sim = beta_d(7, s);
    mu_sim = beta_d(8, s);
    phi_0_sim = beta_d(9, s);
    gamma_1_sim = beta_d(10, s);
    phi_1_sim = beta_d(11, s);
 
    local_sigma_matrix = zeros(length(b_values), length(Q_values));
   
    % Calculate sigmas for each combination of b & q
    for i = 1:length(b_values)
        for j = 1:length(Q_values)
            b_i = b_values(i);
            Q_j = Q_values(j);

            local_sigma_matrix(i, j, s) = calculate_sigma(alpha_0_sim, alpha_1_sim, alpha_2_sim, alpha_3_sim, alpha_4_sim, alpha_5_sim, ...
                                  gamma_0_sim, gamma_1_sim, mu_sim, phi_0_sim, phi_1_sim, theta_hat, ...
                                  W_simulated, Z_simulated, BL, q_simulated, price, N_j);
        end
    end
    sigma_matrices(:,:,s) = local_sigma_matrix; % Store results in sigma_matrices 
end

delete(gcp('nocreate'));   % Close the parallel pool


%% 

% 6. For each new parameter vector, calculate dS/db and dS/dq
dsigma_db_values = zeros(N_sim, length(b_values) - 1, length(Q_values));
dsigma_dq_values = zeros(N_sim, length(b_values), length(Q_values) - 1);

for s = 1:N_sim
    sigma_matrix = sigma_matrices(:, :, s);

    % Partial derivatives w.r.t. b
    for j = 1:length(Q_values)
        for i = 1:(length(b_values) - 1)
            dsigma_db_values(s, i, j) = (sigma_matrix(i + 1, j) - sigma_matrix(i, j)) / (b_values(i + 1) - b_values(i));
        end
    end

    % Partial derivatives w.r.t. q
    for i = 1:length(b_values)
        for j = 1:(length(Q_values) - 1)
            dsigma_dq_values(s, i, j) = (sigma_matrix(i, j + 1) - sigma_matrix(i, j)) / (Q_values(j + 1) - Q_values(j));
        end
    end
end

%% 
% Initialize result matrices for dS/db and dS/dq
dS_db_values = zeros(N_sim, length(b_values) - 1, length(Q_values)); 
dS_dq_values = zeros(N_sim, length(b_values), length(Q_values) - 1);  

for s = 1:N_sim
    mu_temp = mu;  
    theta_hat_temp = theta_hat;  
    gamma_0_temp = gamma_0;  
    gamma_1_temp = gamma_1;  
    phi_0_temp = phi_0;  
    phi_1_temp = phi_1;  

    % Calculate dS/db
    for j = 1:length(Q_values)
        for i = 1:(length(b_values) - 1)
            dS_db_values(:,i,j,s) = (1 / mu_temp) * ((gamma_0_temp + gamma_1_temp * Z_simulated) + ...
                theta_hat_temp * dsigma_db_values(s, i, j));
        end
    end

    % Calculate dS/dq
    for i = 1:length(b_values)
        for j = 1:(length(Q_values) - 1)
            dS_dq_values(:,i,j,s) = (1 / mu_temp) * ((phi_0_temp + phi_1_temp * Z_simulated) + ...
                theta_hat_temp * dsigma_dq_values(s, i, j));
        end
    end
end

%% 

% 7. Sort the N functional values in ascending order.
sorted_dS_db_values = sort(dS_db_values, 1);  % Sort along rows for each column (b value)
sorted_dS_dq_values = sort(dS_dq_values, 1);  % Sort along rows for each column (Q value)

% 8. Calculate the CI from the sorted values for dS/db and dS/dq.
% Initialize CI matrices with updated dimensions
CI_dS_db = zeros(2, length(b_values) - 1);  % Since we calculate differences between consecutive b values
CI_dS_dq = zeros(2, length(Q_values) - 1);  % Since we calculate differences between consecutive Q values

% Calculate 95% CI for dS/db
for col = 1:(length(b_values) - 1)
    CI_dS_db(:, col) = prctile(sorted_dS_db_values(:, col), [2.5, 97.5]);
end

% Calculate 95% CI for dS/dq
for col = 1:(length(Q_values) - 1)
    CI_dS_dq(:, col) = prctile(sorted_dS_dq_values(:, col), [2.5, 97.5]);
end

% Display results
disp('95% Confidence Intervals for dS/db:');
disp(CI_dS_db);

disp('95% Confidence Intervals for dS/dq:');
disp(CI_dS_dq);

%% Final Output Summary
% Display logistic regression coefficients
disp('Logistic Regression Coefficients:');
disp(logit_model);

disp(['Constant (intercept): ', num2str(constant)]);
disp(['Share of pay=1 (sigma): ', num2str(sigma)]);
disp(['Theta_hat (θhat_0): ', num2str(theta_hat)]);

% Display confidence intervals for dS/db
disp('95% Confidence Intervals for dS/db:');
disp(CI_dS_db);

% Display confidence intervals for dS/dq
disp('95% Confidence Intervals for dS/dq:');
disp(CI_dS_dq);



