function NINF_04_ChaoticNoise
% NINF_04_CHAOTICNOISE
% Implements Section 3.3.3: Perturbation using a deterministic chaotic system
% (specifically a Gamma Function approximation in the complex plane).

    %% 1. Configuration
    W = 2048; H = 2048;
    FILENAME = 'Fig4c_GammaChaos.png';
    MAX_ITER = 60;
    
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    % Gamma Function Coefficients (Lanczos approximation)
    C_LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, ...
          771.32342877765313, -176.61502916214059, 12.507343278686905, ...
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    SQRT_2PI = 2.5066282746310005;

    %% 2. Initialization
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(x, y);
    
    C = X + 1i*Y;
    Z = ones(H, W); 
    K = zeros(H, W);
    active = true(H, W);
    
    %% 3. Main Loop
    tic;
    for k = 1:MAX_ITER
        if ~any(active(:)), break; end
        
        z_curr = Z(active);
        
        % --- Step A: "Noise" Calculation (Gamma Function) ---
        % Reflection formula for real(z) < 0.5
        mask_neg = real(z_curr) < 0.5;
        z_calc = z_curr;
        z_calc(mask_neg) = 1 - z_curr(mask_neg);
        
        % Lanczos Approximation
        z_minus_1 = z_calc - 1;
        Acc = C_LANCZOS(1) + C_LANCZOS(2)./(z_minus_1+1) + C_LANCZOS(3)./(z_minus_1+2) + ...
              C_LANCZOS(4)./(z_minus_1+3) + C_LANCZOS(5)./(z_minus_1+4) + ...
              C_LANCZOS(6)./(z_minus_1+5) + C_LANCZOS(7)./(z_minus_1+6) + ...
              C_LANCZOS(8)./(z_minus_1+7) + C_LANCZOS(9)./(z_minus_1+8);
              
        t = z_minus_1 + 7.5;
        GammaVal = SQRT_2PI * (t .^ (z_minus_1 + 0.5)) .* exp(-t) .* Acc;
        
        % Apply reflection adjustment
        GammaVal(mask_neg) = pi ./ (sin(pi * z_curr(mask_neg)) .* GammaVal(mask_neg));
        
        % --- Step B: Distortion ---
        % In this specific case, the "noise" IS added to Z, but it's part of the chaos
        z_dist = z_curr + GammaVal; 
        
        % Safety
        z_dist(abs(z_dist) < 1e-6) = 1e-6;
        
        % --- Step C: Nova Step ---
        % z_new = z_dist - (z_dist^3 - 1) / (3z_dist^2) + c
        Zs = z_dist.^2;
        z_new = z_dist - (z_dist.*Zs - 1)./(3*Zs) + C(active);
        
        % --- Step D: Update ---
        diff = abs(z_curr - z_new);
        converged = diff < 0.005;
        
        Z(active) = z_new;
        if any(converged)
             idx = find(active);
             done = idx(converged);
             K(done) = k;
             active(done) = false;
        end
    end
    K(active) = MAX_ITER; % Fill remainder
    fprintf('Chaos Calculation Time: %.2fs\n', toc);

    %% 4. Rendering
    img = log(K/MAX_ITER + 1);
    img_norm = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    
    if exist('slanCM', 'file'), cmap = slanCM(79); else, cmap = parula(256); end
    rgb = ind2rgb(ceil(img_norm * (size(cmap,1)-1)) + 1, cmap);
    imwrite(rgb, FILENAME);
    imshow(rgb); title('Gamma Function Perturbation');
end