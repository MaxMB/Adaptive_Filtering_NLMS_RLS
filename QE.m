close, clear, clc;
H = [-3, -1, 0, 1, -9, 2, 3, 0, 5, -4];
M = length(H); % filter size

var_u = 1;
var_v = 1e-4;
b = 0.93;

del_NLMS = 1e-6;
mu_NLMS = 0.1;

del_RLS = 1;
lambda_RLS = 1 - mu_NLMS / M;

N = 5000; % iterations
r = 50; % realizations

e2_NLMS_mean = zeros(1,N);
e2_RLS_mean = zeros(1,N);

for i = 1:r
    u = sqrt(1-b^2) * filter([1,-b], 1, sqrt(var_u) * randn(N,1));
    v = sqrt(var_v) * randn(N,1);
    d = filter(H,1,u) + v;
    
    % NLMS filter
    w_NLMS = zeros(1,M); % NLMS coef
    e2_NLMS = zeros(1,N); % NLMS square error
    Nu = 0; % u norm
    for j = 1:N
        if (j > M)
            Nu = Nu + u(j)^2 - u(j-M+1)^2;
            x = flip(u((j-M+1):j));
        else
            Nu = Nu + u(j)^2;
            x = [flip(u(1:j)); zeros(M-j,1)];
        end
        y = w_NLMS * x;
        e = d(j) - y;
        w_NLMS = w_NLMS + e * x' * mu_NLMS / (del_NLMS + Nu);
        e2_NLMS(j) = e^2;
    end
    e2_NLMS_mean = e2_NLMS_mean + e2_NLMS;
    
    % RLS filter
    w_RLS = zeros(1,M); % RLS coef
    e2_RLS = zeros(1,N); % RLS square error
    R_inv = (1/del_RLS) * eye(M);
    for j = 1:N
        if (j > M)
            x = flip(u((j-M+1):j));
        else
            x = [flip(u(1:j)); zeros(M-j,1)];
        end
        k_tilde = R_inv * x;
        gamma_tilde = 1 / (lambda_RLS + x' * k_tilde);
        k = gamma_tilde * k_tilde;
        y = w_RLS * x;
        e = d(j) - y;
        w_RLS = w_RLS + e * k';
        R_inv = (R_inv - gamma_tilde * (k_tilde * k_tilde')) / lambda_RLS;
        e2_RLS(j) = e^2;
    end
    e2_RLS_mean = e2_RLS_mean + e2_RLS;
end
e2_NLMS_mean = 20 * log10( e2_NLMS_mean / r );
e2_RLS_mean = 20 * log10( e2_RLS_mean / r );

% MSE(n) = E{e(n)^2} steady state
MSE_NLMS = mean(e2_NLMS_mean(round(0.95*N):N));
MSE_RLS = mean(e2_RLS_mean(round(0.95*N):N));

figure(1), set(gcf,'color','w');
plot(e2_NLMS_mean,'k'), hold on, plot(e2_RLS_mean,'r'), hold off;
legend('LNMS','RLS'), grid on, xlabel('Iteration [n]'), ylabel('dB');
title(['MSE -- NLMS x RLS -- Realizations = ' num2str(r) ' -- MSE_{NLMS} = ' ...
     num2str(MSE_RLS) ' -- MSE_{RLS} = ' num2str(MSE_NLMS)]);