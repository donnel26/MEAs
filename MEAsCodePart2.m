clear; clc;
%% 
load('MEAsPart1data.mat');
%% 
% 6. For each new parameter vector, calculate dS/db and dS/dq
dsigma_db_values = zeros(length(b_values)-1, length(Q_values), N_sim); 
dsigma_dq_values = zeros(length(b_values), length(Q_values)-1, N_sim); 

for s = 1:N_sim
    sigma_matrix = sigma_matrices(:, :, s);

    % Partial derivatives w.r.t. b
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


%% 
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


%% Final Output Summary
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

%% 
% Display dS/db values for the first simulation and the first Q value
simulation_index = 1;  % Change this to select different simulations
Q_index = 1;           % Change this to select different Q values

% Extract and display the slice
disp(dS_db_values(:, :, Q_index, simulation_index));

