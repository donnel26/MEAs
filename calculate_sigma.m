function sigma = calculate_sigma(alpha_0_sim, alpha_1_sim, alpha_2_sim, alpha_3_sim, alpha_4_sim, alpha_5_sim, ...
                                  gamma_0_sim, gamma_1_sim, mu_sim, phi_0_sim, phi_1_sim, theta_hat, ...
                                  W_simulated, Z_simulated, BL, q_simulated, price, N_j)
    sigma_0 = 0.5; 
    diff = 1;    
    while diff >= 1e-12
        pi_ij = 1 ./ (1 + exp(alpha_1_sim .* W_simulated(:, 1) + alpha_2_sim .* W_simulated(:, 2) + ...
                              alpha_3_sim .* W_simulated(:, 3) + alpha_4_sim .* W_simulated(:, 4) + ...
                              alpha_5_sim .* W_simulated(:, 5)) + theta_hat .* sigma_0 + ...
                              (gamma_0_sim + gamma_1_sim .* Z_simulated) .* BL + ...
                              (phi_0_sim + phi_1_sim .* Z_simulated) .* q_simulated - mu_sim .* price);
        sigma_new = (1 / N_j) * sum(pi_ij);
        diff = max(abs(sigma_new - sigma_0));
        sigma_0 = sigma_new;
    end
    sigma = sigma_0;
end





