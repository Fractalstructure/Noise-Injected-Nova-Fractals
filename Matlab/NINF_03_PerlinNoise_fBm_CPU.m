function NINF_03_PerlinNoise_fBm
% NINF_03_PERLINNOISE_FBM
% Implements Section 3.3.2: Perturbation based on Gradient (Perlin) Noise.
% Generates smooth, marble-like, or fluid textures.

    %% 1. Configuration
    W = 2048; H = 2048;
    FILENAME = 'Fig3b_PerlinNoise.png';
    MAX_ITER = 30;
    
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    ALPHA = 0.35 * 0.8; % Perturbation Strength (Adjusted for Perlin amplitude)

    %% 2. Initialization
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(x, y);
    
    C = X + 1i*Y;
    Z = ones(H, W);           % Start from 1
    K = ones(H, W) * MAX_ITER;
    active = true(H, W);

    %% 3. Main Loop
    fprintf('Running Perlin Noise Perturbation...\n');
    tic;
    for k = 1:MAX_ITER
        if ~any(active(:)), break; end
        
        z_curr = Z(active);
        c_curr = C(active);
        
        % --- Step A: Perturbation (Perlin fBm) ---
        % Inlined loop for 5 octaves
        noiseVal = zeros(size(z_curr));
        amp = 1.0; freq = 1.0;
        
        for o = 1:5
            nx = real(z_curr) * freq;
            ny = imag(z_curr) * freq;
            
            % Gradient Noise Logic
            ix = floor(nx); iy = floor(ny);
            fx = nx - ix;   fy = ny - iy;
            
            % Fade function: 6t^5 - 15t^4 + 10t^3
            ux = fx.^3 .* (fx .* (fx.*6 - 15) + 10);
            uy = fy.^3 .* (fy .* (fy.*6 - 15) + 10);
            
            % Gradients at 4 corners
            n00 = grad_hash(ix,   iy,   fx,   fy);
            n10 = grad_hash(ix+1, iy,   fx-1, fy);
            n01 = grad_hash(ix,   iy+1, fx,   fy-1);
            n11 = grad_hash(ix+1, iy+1, fx-1, fy-1);
            
            % Interpolation
            nx0 = n00 + ux .* (n10 - n00);
            nx1 = n01 + ux .* (n11 - n01);
            n = nx0 + uy .* (nx1 - nx0);
            
            noiseVal = noiseVal + n * amp;
            amp = amp * 0.5;
            freq = freq * 2.0;
        end
        
        % --- Step B: Distortion ---
        z_dist = z_curr + noiseVal * ALPHA;
        
        % --- Step C: Nova Step (Optimized) ---
        % z = 2/3*z + 1/(3z^2) + c
        z_next = (2/3) * z_dist + 1 ./ (3 * z_dist.^2 + 1e-12) + c_curr;
        
        % --- Step D: Convergence ---
        converged = abs(z_next - z_curr) < 0.005;
        
        Z(active) = z_next;
        if any(converged)
             idx = find(active);
             done = idx(converged);
             K(done) = k;
             active(done) = false;
        end
    end
    fprintf('Time: %.2fs\n', toc);

    %% 4. Rendering
    img = log(K) / MAX_ITER;
    img = log(img + 1);
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    
    if exist('slanCM', 'file'), cmap = slanCM(72); else, cmap = parula(256); end
    rgb = ind2rgb(ceil(img_norm * (size(cmap,1)-1)) + 1, cmap);
    imwrite(rgb, FILENAME);
    imshow(rgb); title('Perlin Noise fBm');
end

% --- Helper: Gradient Hash ---
function d = grad_hash(ix, iy, x, y)
    % Simple sin-based hash for gradient direction
    st = sin(ix .* 12.9898 + iy .* 78.233) .* 43758.5453;
    angle = (st - floor(st)) .* 6.283185;
    d = cos(angle) .* x + sin(angle) .* y;
end