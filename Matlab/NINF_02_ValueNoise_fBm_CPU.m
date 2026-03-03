function NINF_02_ValueNoise_fBm
% NINF_02_VALUENOISE_FBM
% Implements Section 3.3.2: Perturbation based on Value Noise (fBm).
% Generates blocky, glacial, or rocky textures.

    %% 1. Configuration
    W = 2048; H = 2048;
    FILENAME = 'Fig3a_ValueNoise.png';
    MAX_ITER = 60;
    
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    ALPHA = 0.35;       % Perturbation Strength
    NOISE_OCTAVES = 5;
    NOISE_LACUNARITY = 3.0;
    NOISE_GAIN = 0.5;

    %% 2. Initialization
    fprintf('Initializing Grid (%dx%d)...\n', W, H);
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(x, y);
    
    C = X + 1i*Y;
    Z = zeros(H, W);        % Start from 0
    K = ones(H, W) * MAX_ITER; 
    active = true(H, W);
    
    %% 3. Main Loop
    tic;
    for k = 1:MAX_ITER
        if ~any(active(:)), break; end
        z_curr = Z(active);
        
        % --- Step A: Perturbation (Value Noise fBm) ---
        noiseVal = value_fbm(z_curr, NOISE_OCTAVES, NOISE_LACUNARITY, NOISE_GAIN);
        
        % --- Step B: Distortion ---
        z_dist = z_curr + noiseVal * ALPHA;
        
        % Safety check
        tiny = abs(z_dist) < 1e-6;
        z_dist(tiny) = 1e-6;
        
        % --- Step C: Nova Step ---
        % z_{n+1} = z - (z^3 - 1) / (3z^2) + c
        z_new = z_dist - (z_dist.^3 - 1) ./ (3 * z_dist.^2) + C(active);
        
        % --- Step D: Update ---
        diff = abs(z_curr - z_new);
        converged = diff < 0.005;
        
        % Update Z
        Z(active) = z_new;
        
        % Handle convergence
        if any(converged)
            act_idx = find(active);
            done_idx = act_idx(converged);
            K(done_idx) = k;
            active(done_idx) = false;
        end
    end
    t_cost = toc;
    fprintf('Finished in %.2fs.\n', t_cost);

    %% 4. Rendering
    % Inverse mapping: Higher iteration = darker (or distinct color)
    img = (MAX_ITER - K) / MAX_ITER; 
    img = log(img + 1);
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    
    if exist('slanCM', 'file'), cmap = slanCM(72); else, cmap = parula(256); end
    rgb = ind2rgb(ceil(img_norm * (size(cmap,1)-1)) + 1, cmap);
    
    imwrite(rgb, FILENAME);
    imshow(rgb); title('Value Noise fBm');
end

% --- Helper: Vectorized Value Noise fBm ---
function total = value_fbm(z, octaves, lacunarity, gain)
    total = zeros(size(z));
    amp = 1; freq = 1;
    rx = real(z); ry = imag(z);
    
    for i = 1:octaves
        x = rx * freq; y = ry * freq;
        ix = floor(x); iy = floor(y);
        fx = x - ix;   fy = y - iy;
        
        % Smoothstep
        u = fx.^2 .* (3 - 2*fx);
        v = fy.^2 .* (3 - 2*fy);
        
        % Value Noise Hash (Pseudo-random)
        n00 = fract(sin(ix*12.9898 + iy*78.233)*43758.5453);
        n10 = fract(sin((ix+1)*12.9898 + iy*78.233)*43758.5453);
        n01 = fract(sin(ix*12.9898 + (iy+1)*78.233)*43758.5453);
        n11 = fract(sin((ix+1)*12.9898 + (iy+1)*78.233)*43758.5453);
        
        % Bilinear Interpolation
        val = (n00 + (n10-n00).*u) + ((n01 + (n11-n01).*u) - (n00 + (n10-n00).*u)).*v;
        
        % Accumulate
        total = total + (val*2-1) * amp;
        freq = freq * lacunarity;
        amp = amp * gain;
    end
    total = (total + 1) / 2;
end

function out = fract(in), out = in - floor(in); end