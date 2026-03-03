function NINF_01_CellNoise
% NINF_01_CELLNOISE
% Implements Section 3.3.1: Perturbation based on Voronoi (Cellular) Noise.
% Generates textures resembling biological tissues or coral structures.

    %% 1. Configuration
    W = 2048; H = 2048;
    FILENAME = 'Fig2_CellNoise.png';
    MAX_ITER = 100;
    
    % Domain settings (Nova Fractal standard view)
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    % Perturbation parameters
    ALPHA = 0.4;        % Perturbation strength
    NUM_SEEDS = 300;    % Number of Voronoi seeds
    SEED_SCALE = 12.0;  % Spread of seeds in complex plane

    %% 2. Initialization
    fprintf('Initializing Grid (%dx%d)...\n', W, H);
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    
    % Create meshgrid
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(single(x), single(y)); % Use single precision for speed
    
    C_grid = complex(X, Y);       % C parameter for Nova
    Z = zeros(H, W, 'single');    % Initial Z = 0 (or 1)
    K = zeros(H, W, 'single');    % Iteration count
    active = true(H, W);          % Mask for non-converged pixels
    
    % Pre-generate Voronoi seeds
    rng(42);
    SEEDS = complex(single((rand(1, NUM_SEEDS)-0.5)*SEED_SCALE), ...
                    single((rand(1, NUM_SEEDS)-0.5)*SEED_SCALE));

    %% 3. Main Loop (Vectorized)
    fprintf('Starting Iteration...\n');
    tic;
    
    for iter = 1:MAX_ITER
        % Get indices of active (non-converged) pixels
        idx = find(active);
        if isempty(idx), break; end
        
        z_curr = Z(idx);
        
        % --- Step A: Calculate Perturbation Field (Voronoi) ---
        % Vectorized distance calculation to all seeds
        n_active = length(idx);
        min_d = inf(n_active, 1, 'single');
        
        % Check distance to every seed (Optimized loop)
        for s = 1:NUM_SEEDS
            d = abs(z_curr - SEEDS(s));
            min_d = min(min_d, d);
        end
        min_d = min(min_d, 1.0); % Clamp distance
        
        % --- Step B: Coordinate Distortion ---
        % z_distorted = z_n + alpha * noise
        z_dist = z_curr + min_d * ALPHA;
        
        % Avoid division by zero
        tiny_mask = abs(z_dist) < 1e-6;
        z_dist(tiny_mask) = 1e-6;
        
        % --- Step C: Nova Iteration Step ---
        % Formula: z_{n+1} = z - (z^3-1)/(3z^2) + c
        % Optimized as: (2z^3 + 1)/(3z^2) + c
        z2 = z_dist .* z_dist;
        z3 = z2 .* z_dist;
        z_new = (2*z3 + 1) ./ (3*z2) + C_grid(idx);
        
        % --- Step D: Convergence Check ---
        diff = abs(z_curr - z_new);
        has_converged = diff < 0.005; % Tolerance
        
        % Update state
        Z(idx) = z_new;
        
        % Mark converged pixels
        if any(has_converged)
            converged_full_idx = idx(has_converged);
            K(converged_full_idx) = iter;
            active(converged_full_idx) = false;
        end
        
        if mod(iter, 10) == 0
             fprintf('Iter: %d/%d, Active: %.2f%%\n', iter, MAX_ITER, 100*length(idx)/numel(active));
        end
    end
    t_cost = toc;
    fprintf('Finished in %.2fs.\n', t_cost);

    %% 4. Rendering
    K(active) = MAX_ITER; % Fill background
    img_log = log(K + 1); % Logarithmic scaling for better contrast
    img_norm = (img_log - min(img_log(:))) / (max(img_log(:)) - min(img_log(:)));
    
    % Colormap selection
    if exist('slanCM', 'file'), cmap = slanCM(75); else, cmap = parula(256); end
    rgb = ind2rgb(ceil(img_norm * (size(cmap,1)-1)) + 1, cmap);
    
    imwrite(rgb, FILENAME);
    imshow(rgb); title(['Cell Noise Perturbation (Time: ' num2str(t_cost) 's)']);
end