set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesFontSize', 15); % Optional: set default font size
set(0, 'DefaultLegendInterpreter', 'latex'); % Set for legends too
close all;

ns=0:40;
even_idx = (mod(ns,2)==0); odd_idx=(mod(ns,2)==1);
ms=zeros(1,length(ns));
for n=ns
    if mod(n,2)==0
        ms(n+1) = nchoosek(n,n/2) / 2^n;
    else
        ms(n+1) = nchoosek(n-1, (n-1)/2) / 2^(n-1);
    end
end

f = -ns.*(ms.^2)/2;
figure; subplot(1,2,1); hold on;
plot(ns(even_idx),ms(even_idx),'--.r','MarkerSize',20);
plot(ns(odd_idx),ms(odd_idx),'--.b','MarkerSize',20);
xlabel("$n$"); ylabel("$m_n$");

subplot(1,2,2); hold on;
plot(ns(even_idx),f(even_idx),'--.r','MarkerSize',20);
plot(ns(odd_idx),f(odd_idx),'--.b','MarkerSize',20);
xlabel("$n$"); ylabel("$f_n$");

%% 
nns = 1:100;
beta = 0.9999999;
ff = - 3*nns*(1-beta)^2 ./ (2*beta^3*(3*nns - 2));
plot(nns, ff);

%% 
alpha = linspace(0.01, 0.3, 1000);
m = linspace(-2,2,30000);
% figure; hold on; plot(m,m,'--k','LineWidth',3);

delta=0.0001; step=0.00001;
m_sol= zeros(length(alpha), 1);
r_sol = zeros(length(alpha), 1);
for i=1:length(alpha)
    f=@(m,r) erf(m/sqrt(2*r*alpha(i)));
    g=@(m,r) 1/((1 - sqrt(2/(pi*r*alpha(i))) * exp(-m^2/(2*r*alpha(i)))))^2;
    m_guess = 1.2; r_guess=1.6;
    while abs(m_guess-f(m_guess,r_guess)) + abs(r_guess-g(m_guess,r_guess))>1e-8
        m_guess = 0.5*f(m_guess, r_guess) + 0.5*m_guess;
        r_guess = 0.5*g(m_guess, r_guess) + 0.5*r_guess;
    end
    m_sol(i)=m_guess; r_sol(i)=r_guess;
    % plot(m, erf(m/sqrt(2*alpha(i)*r_guess)),'LineWidth',2);
end

figure; subplot(1,3,1); plot(alpha, m_sol,'-k','LineWidth',3); 
xlabel("$\alpha$"); ylabel("$m_0$");
subplot(1,3,2); plot(alpha, r_sol,'-k','LineWidth',3);
xlabel("$\alpha$"); ylabel("$r$");
subplot(1,3,3); idx=[4, 400, 442, 452, 600];
hold on; plot(m,m,'--k','LineWidth',3, 'HandleVisibility','off');
for i=idx
    plot(m, erf(m/sqrt(2*alpha(i)*r(i))),'LineWidth',2);
end
xlabel("$m_0$"); ylabel("$m_0$");
legend(strcat('$\alpha$=',string(alpha(idx))));