function NINF_05_IceField_Optimized

    W = 1024; H = 1024;
    FILENAME = 'IceField_Optimized.png';
    MAX_ITER = 60;
    
    Y_MIN = -1.6; Y_MAX = 1.6;
    X_CENTER = -0.4;
    
    ALPHA = 0.30;
    Z_SCALE = 0.08;
    
    NOISE_RES = 2048;
    NOISE_BOUNDS = 8;

    fprintf('1. Pre-calculating Noise Texture (%dx%d)...\n', NOISE_RES, NOISE_RES);
    t_noise = tic;
    
    nx = linspace(-NOISE_BOUNDS, NOISE_BOUNDS, NOISE_RES);
    ny = linspace(-NOISE_BOUNDS, NOISE_BOUNDS, NOISE_RES);
    [NX, NY] = meshgrid(nx, ny);
    NZ = NX + 1i*NY;

    NoiseMap = ice_fbm_vectorized(NZ, 6, 2.5, 0.5);

    map_scale = (NOISE_RES - 1) / (2 * NOISE_BOUNDS);
    map_offset = NOISE_BOUNDS * map_scale + 1;
    
    fprintf('   Noise Texture generated in %.2fs\n', toc(t_noise));

    fprintf('2. Initializing Viewport...\n');
    aspect = W / H;
    x_range = (Y_MAX - Y_MIN) * aspect;
    x = linspace(X_CENTER - x_range/2, X_CENTER + x_range/2, W);
    y = linspace(Y_MIN, Y_MAX, H);
    [X, Y] = meshgrid(x, y);
    
    C = X + 1i*Y;
    Z = zeros(H, W); 
    active = true(H, W);
    K = ones(H, W) * MAX_ITER;

    fprintf('3. Fractal Loop (using Look-Up Table)...\n');
    tic;

    [nm_h, nm_w] = size(NoiseMap);
    
    for k = 1:MAX_ITER-1
        if ~any(active(:)), break; end

        z_curr = Z(active);

        zr = real(z_curr);
        zi = imag(z_curr);

        c_idx = round(zr * map_scale + map_offset);
        r_idx = round(zi * map_scale + map_offset);

        c_idx(c_idx < 1) = 1; c_idx(c_idx > nm_w) = nm_w;
        r_idx(r_idx < 1) = 1; r_idx(r_idx > nm_h) = nm_h;
        
        lin_idx = (c_idx - 1) * nm_h + r_idx;
        noiseVal = NoiseMap(lin_idx);

        z_dist = z_curr + noiseVal * ALPHA;
        
        tiny = abs(z_dist) < 1e-6; 
        z_dist(tiny) = 1e-6;
        
        z2 = z_dist.^2;
        z_new = z_dist - (z_dist .* z2 - 1) ./ (3 * z2) + C(active);
        
        diff = abs(Z(active) - z_new);
        converged = diff < 0.005;

        Z(active) = z_new;
        if any(converged)
             active_indices = find(active);
             just_converged = active_indices(converged);
             
             K(just_converged) = k;
             active(just_converged) = false;
        end
    end
    fprintf('   Calculation finished: %.2fs\n', toc);

    fprintf('4. Rendering...\n');

    H_raw = double(MAX_ITER - K);
    H_norm = (H_raw - min(H_raw(:))) / (max(H_raw(:)) - min(H_raw(:)));
    
    margin = 20;
    H_crop = H_norm(margin:end-margin, margin:end-margin);
    X_crop = X(margin:end-margin, margin:end-margin);
    Y_crop = Y(margin:end-margin, margin:end-margin);
    
    if exist('imgaussfilt', 'file')
        H_smooth = imgaussfilt(H_crop, 2.0);
    else
        h_gauss = fspecial('gaussian', [9 9], 2.0);
        H_smooth = imfilter(H_crop, h_gauss);
    end
    
    H_final = H_smooth * 0.9 + H_crop * 0.1;
    Z_elevation = H_final * Z_SCALE;

    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 1000]);
    
    s = surf(X_crop, Y_crop, Z_elevation);
    s.EdgeColor = 'none';
    s.FaceColor = 'interp'; 

    ice_cmap = [0.02, 0.10, 0.25; 0.15, 0.35, 0.55; 0.60, 0.75, 0.85; 0.95, 0.98, 1.00];
    colormap(interp1([0, 0.3, 0.7, 1], ice_cmap, linspace(0,1,512)));
    
    lighting phong;
    material([0.3, 0.6, 1, 20]); 
    
    view(-30, 55);
    axis off; axis vis3d; axis tight;
    camzoom(1.2);

    light('Position', [-1, -1, 2], 'Style', 'infinite', 'Color', [1, 0.95, 0.9]); 
    light('Position', [1, 1, 0.5], 'Style', 'infinite', 'Color', [0.3, 0.4, 0.6]); 
    
    daspect([1 1 0.4]); 
    
    fprintf('Done. Saving to %s\n', FILENAME);
end

function total = ice_fbm_vectorized(z, octaves, lacunarity, gain)
    total = zeros(size(z));
    amp = 1; 
    freq = 1;
    
    rx = real(z); 
    ry = imag(z);
    
    for i = 1:octaves
        x = rx * freq; 
        y = ry * freq;
        
        ix = floor(x); 
        iy = floor(y);
        
        fx = x - ix;   
        fy = y - iy;
        
        u = fx.^2 .* (3 - 2*fx); 
        v = fy.^2 .* (3 - 2*fy);

        n00 = fract_vec(sin(ix*12.9898 + iy*78.233)*43758.5453);
        n10 = fract_vec(sin((ix+1)*12.9898 + iy*78.233)*43758.5453);
        n01 = fract_vec(sin(ix*12.9898 + (iy+1)*78.233)*43758.5453);
        n11 = fract_vec(sin((ix+1)*12.9898 + (iy+1)*78.233)*43758.5453);

        nx0 = n00 + (n10 - n00) .* u;
        nx1 = n01 + (n11 - n01) .* u;
        val = nx0 + (nx1 - nx0) .* v;
        
        total = total + (val*2-1) * amp;
        
        freq = freq * lacunarity; 
        amp = amp * gain;
    end
    total = (total + 1) / 2;
end

function out = fract_vec(in)
    out = in - floor(in);
end