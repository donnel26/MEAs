delete(gcp('nocreate'));   % this line closes the parallel pool


clear; clc;
rng(1001); 
Ngene_df = readtable('C:\\Users\\donnel26\\OneDrive - purdue.edu\\RA assignment FA ''23\\SHC\\NgeneSimMEAs.csv');

% Parameters
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
prob = Ngene_df.yes_prob;        
price = Ngene_df.yes_price_;     
BL_prob = Ngene_df.yes_bl_yes_prob; 
price_prob = Ngene_df.yes_price__yes_prob; 

% Simulating W data 
num_samples = height(Ngene_df);
num_W_vars = 5; % Number of W variables to simulate

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

 %%%%%%%%%%%%%%%%%%%%%%%%
q_simulated = randi([0, 1], num_samples, 1) * 200 + 1800;  % 1800 or 2000

%% 

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
U_yes = (alpha_0 + alpha_1 * W_simulated(:, 1) + alpha_2 * W_simulated(:, 2) +  alpha_3 * W_simulated(:, 3) + alpha_4 * W_simulated(:, 4) + alpha_5 * W_simulated(:, 5)) + (gamma_1 * BL_prob + b_price * price_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E1;
U_no =   (alpha_0 + alpha_1 * W_simulated(:, 1) + alpha_2 * W_simulated(:, 2) +  alpha_3 * W_simulated(:, 3) + alpha_4 * W_simulated(:, 4) + alpha_5 * W_simulated(:, 5)) + (gamma_1 * BL_prob) + (phi_0 + phi_1 * Z_simulated) .* q_simulated + E0;
% Decision variable
D = zeros(height(Ngene_df), 1);  % Initialize D
D(U_yes > U_no) = 1; % Assign 1 if U_yes > U_no
cons = zeros(height(Ngene_df), 1);

share = splitapply(@mean, D, permit_dummy);

% Add the share column back to the dataset
Ngene_df.share = share(permit_dummy);

% estimate logit model
[logit_model, deviance, stats]= glmfit(X, D, 'binomial', 'link', 'logit');

param_names = {'Intercept', 'W1', 'W2', 'W3', 'W4', 'W5', 'BL', 'Price', 'q_simulated', 'bZ_simulated', 'qZ_simulated'}; 

for i = 1:length(param_names) 
   fprintf('%-18s %-12.4f %-12.4f\n', param_names{i}, logit_model(i), stats.se(i)); 
end

for i = 1:length(param_names) 
    % Calculate the confidence intervals
   lower_bound = logit_model(i) - 1.96 * stats.se(i); % Lower bound
   upper_bound = logit_model(i) + 1.96 * stats.se(i); % Upper bound

    % Display the results
% total_obs   fprintf('%-18s %-12.4f %-12.4f [%-.4f, %.4f]\n', param_names{i}, logit_model(i), stats.se(i), lower_bound, upper_bound); 
end

% Extract covariance matrix
[~, ~, stats] = glmfit(X, D, 'binomial', 'link', 'logit');
covariance_matrix = stats.covb;


%save coefficients
alpha_0 = logit_model(1)
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
theta_hat = constant / sigma;

% Drop Duplicates 
[~, ia] = unique(Ngene_df.ChoiceSituation, 'stable'); %% Identifies the first occurence of unique ChoiceSituation
unique = Ngene_df(ia, :); %% Creates a new table, but just with unique ChoiceSituaitons

% Krinsky Robb Stuff
beta_hat = [alpha_0; alpha_1; alpha_2; alpha_3; alpha_4; alpha_5; gamma_0; mu; phi_0; gamma_1; phi_1];
variable_matrix = [W_simulated, BL, price, q_simulated, bZ_simulated, qZ_simulated];
C = chol(covariance_matrix, 'lower');
N_sim = 100; % number of simulations
K = length(beta_hat); % number of params
xK = randn(K, N_sim);
beta_hat = repmat(beta_hat, 1, N_sim);
beta_d = beta_hat + C * xK;
b_values = [1, 2, 3];

Q_values = [2000, 1800]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_j = height(Ngene_df); 

sigma_matrices = zeros(length(b_values), length(Q_values), N_sim);
pi_ij_matrices = zeros(N_j, length(b_values), length(Q_values), N_sim);


% Initialize Parallel Pool
parpool(4);
tic

% Parallel Simulation
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

    % Initialize local_sigma_matrix as a 2D array
    local_sigma_matrix = zeros(length(b_values), length(Q_values));
    local_pij_matrix = zeros(N_j, length(b_values), length(Q_values));
   
    % Calculate sigmas for each combination of b & q
    for i = 1:length(b_values)
        for j = 1:length(Q_values)
            BL = b_values(i);
            q_simulated = Q_values(j);
diff=1
sigma_0=.1
k=.6
while diff >= 1e-12
        pi_ij = 1 ./ (1 + exp(alpha_1_sim .* W_simulated(:, 1) + alpha_2_sim .* W_simulated(:, 2) + ...
                              alpha_3_sim .* W_simulated(:, 3) + alpha_4_sim .* W_simulated(:, 4) + ...
                              alpha_5_sim .* W_simulated(:, 5)) + theta_hat .* sigma_0 + ...
                              (gamma_0_sim + gamma_1_sim .* Z_simulated) .* BL + ...
                              (phi_0_sim + phi_1_sim .* Z_simulated) .* q_simulated - mu_sim .* price);
        sigma_new = sum(pi_ij)/N_j*(1-k)+k*sigma_0;
        diff = max(abs(sigma_0-sigma_new));
        sigma_0 = sigma_new;
end
local_pij_matrix(:,i,j) = pi_ij;
local_sigma_matrix(i,j) = 1/N_j * sum(pi_ij);
         
        end
    end

    % Store the results for this simulation in sigma_matrices
    sigma_matrices(:,:,s) = local_sigma_matrix; 
    pi_ij_matrices(:,:,:,s) = local_pij_matrix; 
end
toc
% Close the parallel pool
delete(gcp('nocreate'));

dsigma_db_values = zeros(length(b_values)-1, length(Q_values), N_sim); 
dsigma_dq_values = zeros(length(b_values), length(Q_values)-1, N_sim); 


for s = 1:N_sim
    sigma_matrix = sigma_matrices(:, :, s);

    % Partial derivatives w.r.t. b Change b= 3 to 2 and b= 2 to 1
    for j = 1:length(Q_values)  % For each Q value
        for i = 1:(length(b_values) - 1)  % For each change in b
            dsigma_db_values(i, j, s) = (sigma_matrix(i + 1, j) - sigma_matrix(i, j)) / (b_values(i + 1) - b_values(i));
        end
    end

    % Partial derivatives w.r.t. q
    for i = 1:length(b_values)  % For each b value
        for j = 1:(length(Q_values) - 1)  % For each change in Q
            dsigma_dq_values(i, j, s) = (sigma_matrix(i, j + 1) - sigma_matrix(i, j)) / (Q_values(j + 1) - Q_values(j));
        end
    end
end

dS_db_values = zeros(N_j, length(b_values)-1, length(Q_values), N_sim); 
dS_dq_values = zeros(N_j, length(b_values), length(Q_values)-1, N_sim);  

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
            dS_db_values(:, i, j, s) = (1 / mu_temp) * ((gamma_0_temp + gamma_1_temp * Z_simulated) + ...
                theta_hat_temp * dsigma_db_values(i, j, s));
        end
    end
% Calculate dS/dq
    for i = 1:length(b_values)
        for j = 1:(length(Q_values) - 1)
            dS_dq_values(:, i, j, s) = (1 / mu_temp) * ((phi_0_temp + phi_1_temp * Z_simulated) + ...
                theta_hat_temp * dsigma_dq_values(i, j, s));
        end
    end
end


% 7. Sort the N functional values in ascending order.
sorted_dS_db_values = sort(dS_db_values, 1);  % Sort along rows for each column (b value)
sorted_dS_dq_values = sort(dS_dq_values, 1);  % Sort along rows for each column (Q value)

% 8. Calculate the CI from the sorted values for dS/db and dS/dq.
% Initialize CI matrices with updated dimensions
CI_dS_db = zeros(2, length(b_values) - 1);  % Since we calculate differences between consecutive b values
CI_dS_dq = zeros(2, length(Q_values) - 1);  % Since we calculate differences between consecutive Q values

% Calculate 95% CI for dS/db
for col = 1:(length(b_values) - 1)
    CI_dS_db(:, col) = prctile(sorted_dS_db_values(:, col, :), [2.5, 97.5], 'all');
end

% Calculate 95% CI for dS/dq
for col = 1:(length(Q_values) - 1)
    CI_dS_dq(:, col) = prctile(sorted_dS_dq_values(:, col, :), [2.5, 97.5], 'all');
end

% Initialize matrices for summary statistics
stats_dS_db = zeros(3, length(b_values) - 1, 2); % Now accounts for two changes in b
stats_dS_dq = zeros(3, length(Q_values) - 1);

% Compute summary statistics for dS/db
for col = 1:(length(b_values) - 1)
    stats_dS_db(1, col, :) = [mean(sorted_dS_db_values(:, col, 1), 'all'), mean(sorted_dS_db_values(:, col, 2), 'all')];  % Two Means
    stats_dS_db(2, col, :) = [prctile(sorted_dS_db_values(:, col, 1), 25, 'all'), prctile(sorted_dS_db_values(:, col, 2), 25, 'all')];  % Two 25th percentiles
    stats_dS_db(3, col, :) = [prctile(sorted_dS_db_values(:, col, 1), 75, 'all'), prctile(sorted_dS_db_values(:, col, 2), 75, 'all')];  % Two 75th percentiles
end

% Compute summary statistics for dS/dq
for col = 1:(length(Q_values) - 1)
    stats_dS_dq(1, col) = mean(sorted_dS_dq_values(:, col), 'all');  % Mean
    stats_dS_dq(2, col) = prctile(sorted_dS_dq_values(:, col), 25, 'all');  % 25th percentile
    stats_dS_dq(3, col) = prctile(sorted_dS_dq_values(:, col), 75, 'all');  % 75th percentile
end


% Display results
disp('95% Confidence Intervals for dS/db:');
disp(CI_dS_db);

disp('95% Confidence Intervals for dS/dq:');
disp(CI_dS_dq);

% Display summary statistics
disp('Summary Statistics for dS/db (Two changes in b):');
disp('Row 1: Mean, Row 2: 25th Percentile, Row 3: 75th Percentile');
disp(stats_dS_db);

disp('Summary Statistics for dS/dq:');
disp('Row 1: Mean, Row 2: 25th Percentile, Row 3: 75th Percentile');
disp(stats_dS_dq);

% Display logistic regression coefficients
disp('Logistic Regression Coefficients:');
disp(logit_model);

disp(['Constant (intercept): ', num2str(constant)]);
disp(['Share of pay=1 (sigma): ', num2str(sigma)]);
disp(['Theta_hat (θhat_0): ', num2str(theta_hat)]);
1;           % Change this to select different Q values

% Extract and display the slice
%disp(dS_db_values(:, :, Q_index, simulation_index));

