Nv = 5; % "Hidden" neurons
Nh = 2; % "Visible" neurons

v = ones(Nv,1); v(randperm(Nv,round(Nv/2))) = -1;
h = rand(Nh,1); h(randperm(Nh,round(Nh/2))) = -1;

W = 2*rand(Nv, Nh)-1; % Weight matrix   
a = 2*rand(Nv,1) - 1; % Bias for visible neurons    
b = 2*rand(Nh,1) - 1; % Bias for hidden neurons

[v_samp, h_samp] = sample_rbm(W,v,h,a,b,10)

function [W_fin, a_fin, b_fin] = train_rbm(W_init, v,h,a_init,b_init);
    
end

function [v_samp, h_samp] = sample_rbm(W, v_init, h_init, a,b,k)
    Nh=length(h_init); Nv=length(v_init);
    v_samp=v_init; h_samp=h_init;
    for iter=1:k
        h_samp=sample_h(W,v_samp,h_samp,a,b);
        v_samp=sample_v(W,v_samp,h_samp,a,b);
    end
end

function h_samp = sample_h(W,v,h_init,a,b);
    h_samp=h_init; Nh=length(h_init);
    for j=1:Nh
        m = sum(v.*W(:,j)) + b(j); % effective field
        p_p1 = exp(m) / (exp(m) + exp(-m));
        if p_p1>=0.5
            h_samp(j) = 1;
        else
            h_samp(j) = -1;
        end
    end
end

function v_samp = sample_v(W,v_init,h,a,b)
    v_samp = v_init; Nv=length(v_init);
    for i=1:Nv
        m=sum(W(i,:).*h) + a(i);
        p_p1 = exp(m) / (exp(m)+exp(-m));
        if p_p1>=0.5
            v_samp(i) = 1;
        else
            v_samp(i) = -1;
        end
    end
end

function E = energy(W,v,h,a,b,normalize)
    E = -sum(v.*(W*h)) - sum(a.*v) - sum(b.*h);
    if nargin==6 && normalize
        E = E / (length(v) + length(h));
    end
end