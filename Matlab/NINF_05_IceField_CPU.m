function NINF_05_IceField_Application
% NINF_05_ICEFIELD_APPLICATION
% Implements Section 4: Procedural generation of Polar Ice Terrain.
% Converts fractal convergence data to heightmaps and renders in 3D.

    %% 1. Configuration (High-Res for Publication)
    W = 2048; H = 2048;
    FILENAME = 'Fig6_IceField.png';
    MAX_ITER = 60;
    
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    ALPHA = 0.30;       % Perturbation Strength
    Z_SCALE = 0.25;     % Vertical terrain scaling factor (flat ice sheet)

    %% 2. Initialization
    fprintf('1. Initializing Terrain Data...\n');
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(x, y);
    
    C = X + 1i*Y;
    Z = zeros(H, W); 
    active = true(H, W);
    K = ones(H, W) * MAX_ITER;

    %% 3. Main Loop (Fractal Generation)
    tic;
    for k = 1:MAX_ITER-1
        if ~any(active(:)), break; end
        z_curr = Z(active);
        
        % Use fBm Noise (Octaves=6, Lacunarity=2.5) for sharper ice details
        noiseVal = ice_fbm(z_curr, 6, 2.5, 0.5); 
        
        % Distort
        z_dist = z_curr + noiseVal * ALPHA;
        
        % Nova Update
        tiny = abs(z_dist) < 1e-6; z_dist(tiny) = 1e-6;
        z_new = z_dist - (z_dist.^3 - 1) ./ (3 * z_dist.^2) + C(active);
        
        % Convergence
        diff = abs(Z(active) - z_new);
        converged = diff < 0.005;
        
        Z(active) = z_new;
        if any(converged)
             idx = find(active);
             done = idx(converged);
             K(done) = k;
             active(done) = false;
        end
    end
    fprintf('Fractal calculation finished: %.2fs\n', toc);

    %% 4. Heightmap Processing (Section 4.2)
    fprintf('2. Processing Heightmap...\n');
    
    % Raw Height: Faster convergence = Higher ground (Ice surface)
    H_raw = double(MAX_ITER - K);
    H_norm = (H_raw - min(H_raw(:))) / (max(H_raw(:)) - min(H_raw(:)));
    
    % Crop edges to remove artifacts
    margin = 20;
    H_crop = H_norm(margin:end-margin, margin:end-margin);
    X_crop = X(margin:end-margin, margin:end-margin);
    Y_crop = Y(margin:end-margin, margin:end-margin);
    
    % Post-processing: Smooth snow over sharp ice
    H_smooth = imgaussfilt(H_crop, 2.0); 
    
    % Blend: 90% smooth (snow) + 10% raw (cracks)
    H_final = H_smooth * 0.9 + H_crop * 0.1;
    Z_elevation = H_final * Z_SCALE;

    %% 5. 3D Rendering (Section 4.3)
    fprintf('3. Rendering 3D Scene...\n');
    
    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 1000]);
    
    % Surface Plot
    s = surf(X_crop, Y_crop, Z_elevation);
    s.EdgeColor = 'none';
    s.FaceColor = 'interp'; % Gouraud/Phong interpolation
    
    % Material & Lighting
    % Custom Ice Colormap: Deep Blue -> Frost -> Snow White
    ice_cmap = [0.02, 0.10, 0.25; 0.15, 0.35, 0.55; 0.60, 0.75, 0.85; 0.95, 0.98, 1.00];
    colormap(interp1([0, 0.3, 0.7, 1], ice_cmap, linspace(0,1,512)));
    
    lighting phong;              % Phong shading for specular highlights
    material([0.9, 0.6, 0.8, 20]); % Ambient, Diffuse, Specular(High), Shininess
    
    % Camera & Lights
    view(-30, 55);
    axis off; axis vis3d; axis tight;
    camzoom(1.2); % Zoom in
    
    % Sun and Fill light
    light('Position', [-1, -1, 2], 'Style', 'infinite', 'Color', [1, 0.98, 0.9]);
    light('Position', [1, 1, 0.5], 'Style', 'infinite', 'Color', [0.4, 0.5, 0.7]);
    
    % Aspect ratio correction
    daspect([1 1 0.4]); 
    
    %% 6. Export
    fprintf('Saving high-res render to %s...\n', FILENAME);
    set(fig, 'InvertHardcopy', 'off', 'PaperPositionMode', 'auto');
    print(fig, FILENAME, '-dpng', '-r300', '-opengl');
end

% --- Helper: fBm (Same logic as Value noise but different parameters) ---
function total = ice_fbm(z, octaves, lacunarity, gain)
    total = zeros(size(z));
    amp = 1; freq = 1;
    rx = real(z); ry = imag(z);
    for i = 1:octaves
        x = rx * freq; y = ry * freq;
        ix = floor(x); iy = floor(y);
        fx = x - ix;   fy = y - iy;
        u = fx.^2 .* (3 - 2*fx); v = fy.^2 .* (3 - 2*fy);
        
        n00 = fract(sin(ix*12.9898 + iy*78.233)*43758.5453);
        n10 = fract(sin((ix+1)*12.9898 + iy*78.233)*43758.5453);
        n01 = fract(sin(ix*12.9898 + (iy+1)*78.233)*43758.5453);
        n11 = fract(sin((ix+1)*12.9898 + (iy+1)*78.233)*43758.5453);
        
        val = (n00+(n10-n00).*u) + ((n01+(n11-n01).*u)-(n00+(n10-n00).*u)).*v;
        total = total + (val*2-1) * amp;
        freq = freq * lacunarity; amp = amp * gain;
    end
    total = (total + 1) / 2;
end

function out = fract(in), out = in - floor(in); end