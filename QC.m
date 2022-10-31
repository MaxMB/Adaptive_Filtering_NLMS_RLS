close, clear, clc;
H = [-3, -1, 0, 1, -9, 2, 3, 0, 5, -4];
M = length(H); % filter size

var_u = 1;
var_v = 1e-4;
b = 0.7;

del_NLMS = 1e-6;
mu_NLMS = 0.1;

del_RLS = 1;
lambda_RLS = 1 - mu_NLMS / M;

N = 5000; % iterations
r = 50; % realizations
w_NLMS_mean = zeros(N,M); % NLMS coef mean
w_RLS_mean = zeros(N,M); % RLS coef mean

for i = 1:r
    u = sqrt(1-b^2) * filter([1,-b], 1, sqrt(var_u) * randn(N,1));
    v = sqrt(var_v) * randn(N,1);
    d = filter(H,1,u) + v;
    
    % NLMS filter
    w_NLMS = zeros(N,M); % NLMS coef
    Nu = 0; % u norm
    for j = 1:(N-1)
        if (j > M)
            Nu = Nu + u(j)^2 - u(j-M+1)^2;
            x = flip(u((j-M+1):j));
        else
            Nu = Nu + u(j)^2;
            x = [flip(u(1:j)); zeros(M-j,1)];
        end
        y = w_NLMS(j,:) * x;
        e = d(j) - y;
        w_NLMS(j+1,:) = w_NLMS(j,:) + e * x' * mu_NLMS / (del_NLMS + Nu);
    end
    w_NLMS_mean = w_NLMS_mean + w_NLMS;
    
    % RLS filter
    w_RLS = zeros(N,M); % RLS coef
    R_inv = (1/del_RLS) * eye(M);
    for j = 1:(N-1)
        if (j > M)
            x = flip(u((j-M+1):j));
        else
            x = [flip(u(1:j)); zeros(M-j,1)];
        end
        k_tilde = R_inv * x;
        gamma_tilde = 1 / (lambda_RLS + x' * k_tilde);
        k = gamma_tilde * k_tilde;
        y = w_RLS(j,:) * x;
        e = d(j) - y;
        w_RLS(j+1,:) = w_RLS(j,:) + e * k';
        R_inv = (R_inv - gamma_tilde * (k_tilde * k_tilde')) / lambda_RLS;
    end
    w_RLS_mean = w_RLS_mean + w_RLS;
end
w_NLMS_mean = w_NLMS_mean / r;
w_RLS_mean = w_RLS_mean / r;

figure(1), set(gcf,'color','w');
for i = 1:M
    subplot(2,5,i);
    plot(w_NLMS_mean(:,i),'k'), hold on;
    plot(w_RLS_mean(:,i),'r'), hold off;
    xlim([0,N]), grid on;
    title(['H(' num2str(i-1) ') = ' num2str(H(i))]);
end
legend('NLMS','RLS');