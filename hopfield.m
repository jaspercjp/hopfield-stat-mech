set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesFontSize', 15); % Optional: set default font size
set(0, 'DefaultLegendInterpreter', 'latex'); % Set for legends too

%% Initialize random Hopfield networks and plot their energies during their 
% training process 
figure; hold on;
N = 500; % Number of neurons 
UPDATES_ITR = 2500;
for i=1:10
    [~,E] = hop_rand(N, UPDATES_ITR);
    plot(E);
end
xlabel("Update steps");
ylabel("Energy per neuron");

%% An example of training the network to memorize an image
im_arr = double(read_pixel_im("smiley-brow.png"));
W_trained = im_arr(:) * im_arr(:)';
s_init = corrupt_arr(im_arr(:), 0.4);
[s_fin, E] = hop_init(s_init, W_trained, UPDATES_ITR);
figure; show_im(s_init, size(im_arr));
figure; show_im(s_fin, size(im_arr));

%% How many memories can the network remember?
N_NEURON = 50;
N_SET = 5;
N_TRIAL = 20;
N_ITR = 50*N_NEURON;
n_mems = 10:5:100;
ps = linspace(0,1,30);
hamming_matrix = zeros(length(n_mems), length(ps));

for i=1:length(ps)
    for j=1:length(n_mems)
        hamming_data = rand_mem_assess(N_SET, N_TRIAL, N_NEURON, n_mems(j), ps(i), N_ITR);
        hamming_matrix(j,i) = mean(hamming_data,'all');
    end
    fprintf("%d/%d Done.\n", i, length(ps))
end

%% Plot the obtained Hamming distances 
figure; pcolor(ps,n_mems,hamming_matrix/N_NEURON);
set(gca, 'YDir', 'normal');
cb = colorbar;
shading interp;
title(sprintf("$N={%d}$",N_NEURON));
xlabel("$\kappa$"); ylabel("$p$"); 
ylabel(cb, "Normalized Hamming Distance",'interpreter','latex');

% Randomly generate binary strings and train the Hopfield network on them.
% Then randomly perturb these strings with probability p, and see if the 
% network can correctly retrieve the original string. 
function hamming_data = rand_mem_assess(n_set, n_trial, n_neuron, ...
    n_mem, p, N_ITR)
    hamming_data = zeros(n_set, n_trial);
    for k=1:n_set % Use sets for randomness
        s_mem = ones(n_neuron, n_mem); 
        s_mem(randperm(numel(s_mem),round(numel(s_mem)/2))) = -1;
        W_trained = zeros(n_neuron, n_neuron);
        % Construct weight matrix based on examples
        for i=1:n_mem
            W_trained = W_trained + s_mem(:,i)*s_mem(:,i)'/n_mem;
        end
        % Do n_trial different perturbation-retrieval attempts 
        for i=1:n_trial
            s_init = corrupt_arr(s_mem(:,randi(n_mem)), p);
            [s_fin, ~] = hop_init(s_init, W_trained, N_ITR);
            hamming_data(k,i) = hamming(s_init, s_fin);
        end
    end
end

function d = hamming(arr1, arr2)
    d = sum(arr1~=arr2);
end

% Flip the sign of p*length(arr) elements in arr. 
function out = corrupt_arr(arr,p)
    n = round(p*numel(arr));
    out = arr;
    indices = randperm(numel(arr),n);
    out(indices) = -out(indices);
end

% ================= FUNCTIONS FOR READING IN IMAGES ===================
function arr = read_pixel_im(filename)
    [~,~,alpha] = imread(filename);
    arr = 2*cast(alpha/255, 'int8') - 1;
end

function show_im(arr, shape)
    scaled_im = 255*((arr+1)/2);
    if nargin == 1
        imshow(scaled_im,'InitialMagnification','fit');
    else
        imshow(reshape(scaled_im,shape), ...
            'InitialMagnification','fit');
    end
end

% ===================== CODE FOR HOPFIELD NETWORK ======================

% Initialize a hopfield network with random states and weights and update it 
% n_itr times
function [s,E] = hop_rand(N, n_itr)
    s = ones(N,1); s(randperm(N,round(N/2))) = -1; % Initiate the states with pm1
    b = 2*rand(N,1) - 1; % Bias vector 
    aux_mat = rand(N); W=aux_mat + aux_mat'; % Symmetric connection matrix, W_ij is the synapse strength between i and j
    
    E = zeros(n_itr, 1); % energy per neuron
    for i=1:n_itr
        s = update(s, b, W);
        E(i) = energy(s,b,W)/N;
        display(abs(E(i)+N/2))
    end
end

% Initialize a Hopfield network with given weights and states and update
% n_itr times
function [final_s,E] = hop_init(s,W, n_itr) 
    [N, ~] = size(s);
    b = 2*rand(N,1) - 1; % Bias vector
    itr = 1;
    E_mem = inf;
    % E_max = N*max(abs(W),[],'all')/2;
    E = zeros(n_itr, 1); % energy per neuron
    stag_count = 0; STAG_THRESHOLD=50;
    while stag_count<STAG_THRESHOLD && itr<n_itr
        s = update(s, b, W);
        E(itr) = energy(s,b,W)/N;
        if abs(E(itr)-E_mem)/abs(E_mem) < 1e-7
            stag_count = stag_count + 1;
        else
            stag_count = 0;
        end
        E_mem = E(itr);
        itr = itr+1;
    end
    final_s = s;
    % display(E_mem)
    % display(E_max)
    % disp((E_mem+E_max)/E_max)
end

% Helper function to update the state of a Hopfield network
function s_np1 = update(s_n, b, W)
    s_np1 = s_n;
    j = randi([1,length(s_n)]); % Randomly choose a neuron to update
    w = W(:,j);
    w(j)=0; 
    s_n(j)=0;
    if dot(w, s_n)>b(j)
        s_np1(j) = 1;
    else
        s_np1(j) = -1;
    end
end

% Computes the energy of a Hopfield network
function E = energy(s,b,W)
    E = -dot(W*s, s)/2 + dot(b,s);
end



