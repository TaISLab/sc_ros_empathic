function plot_paths_offline(csvfile, window, arrow_dt)
%PLOT_PATHS_OFFLINE  Traced paths + sampled velocity arrows from a trial CSV.
%   In the circle plane: the reference circle, the robot end-effector path,
%   and -- every ARROW_DT seconds (default 1) -- arrows for
%       v_h   (human intent)          + eta_h * v_h   overlaid
%       v_r   (path follower)         + eta_r * v_r    overlaid
%       v_h + v_r  (naive sum)        + v_s            overlaid  (the
%                                       eta-weighted emergent command,
%                                       as logged: after eta_s, the LPF
%                                       and the speed cap)
%   Raw = solid, eta-weighted = dashed, same colour per channel.
%
%   plot_paths_offline(csvfile)
%   plot_paths_offline(csvfile, [t0 t1])       % seconds from start
%   plot_paths_offline(csvfile, N)             % just lap N (scalar)
%   plot_paths_offline(csvfile, sel, arrow_dt) % arrow spacing, s
%
%   Geometry (circle centre / radius / plane normal) is read from a
%   "<csv>.params.json" sidecar if present, else a horizontal circle at
%   [0.45 0 0.45] with radius 0.05 (the launch defaults).
%
%   Requires MATLAB R2019b+.

    if nargin < 2, window = []; end
    if nargin < 3 || isempty(arrow_dt), arrow_dt = 1.0; end

    [C, R, Nrm, geom_src] = load_sidecar_geom(csvfile);

    T = readtable(csvfile);
    t = T.t_rel - T.t_rel(1);
    lap_only = [];
    if isempty(window)
        sel = true(height(T),1);
    elseif isscalar(window)
        lap_only = window;  sel = T.lap == window;
    else
        sel = t >= window(1) & t <= window(2);
    end
    assert(any(sel), 'No rows in the requested window / lap.');

    % ---- circle-plane orthonormal basis (u, w), origin at C -------
    n = Nrm(:) / norm(Nrm);
    a = [1;0;0]; if abs(n.'*a) > 0.9, a = [0;1;0]; end
    u = a - (a.'*n)*n;  u = u / norm(u);
    w = cross(n, u);
    p2 = @(M) [ (M - C(:).') * u , (M - C(:).') * w ];   % rows -> Nx2
    v2 = @(M) [ M * u , M * w ];                          % vectors -> Nx2

    xy = p2([T.px(sel) T.py(sel) T.pz(sel)]);            % robot path
    tf = t(sel);
    Vh  = v2([T.vh_x(sel) T.vh_y(sel) T.vh_z(sel)]);
    Vr  = v2([T.vr_x(sel) T.vr_y(sel) T.vr_z(sel)]);
    Vs  = v2([T.vs_x(sel) T.vs_y(sel) T.vs_z(sel)]);
    eh  = T.eta_h(sel);  er = T.eta_r(sel);
    Vsum = Vh + Vr;
    Vhw  = eh .* Vh;   Vrw = er .* Vr;                   % eta-weighted

    th  = linspace(0, 2*pi, 240);
    ref = R * [cos(th); sin(th)];

    % ---- arrow sample points (~ every arrow_dt s) ----------------
    idx = 1; last = tf(1);
    for k = 2:numel(tf)
        if tf(k) - last >= arrow_dt, idx(end+1) = k; last = tf(k); end %#ok<AGROW>
    end
    P0 = xy(idx,:);
    ref_mag = median(vecnorm(Vsum(idx,:), 2, 2));
    g = 0.6 * R / max(ref_mag, 1e-4);                    % m of arrow per m/s

    % ---- figure -------------------------------------------------
    figure('Name','paths + velocity arrows','Color','w', ...
           'Position',[120 90 820 760]);
    ax = axes; hold(ax,'on'); axis(ax,'equal'); grid(ax,'on'); box(ax,'on');
    plot(ax, ref(1,:), ref(2,:), '--', 'Color',[.55 .55 .55], 'LineWidth',1.3);
    plot(ax, xy(:,1), xy(:,2), '-', 'Color',[.1 .1 .1], 'LineWidth',1.0);
    plot(ax, xy(1,1), xy(1,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize',5);

    GH = [0 .60 0]; BL = [.15 .35 .90]; GY = [.35 .35 .35]; RD = [.85 .10 .10];
    q = @(V, c, ls, lw) quiver(ax, P0(:,1), P0(:,2), ...
            g*V(idx,1), g*V(idx,2), 0, 'Color',c, 'LineStyle',ls, ...
            'LineWidth',lw, 'MaxHeadSize',0.35);
    hHr = q(Vh,   GH, '-',  1.3);
    hRr = q(Vr,   BL, '-',  1.3);
    hSr = q(Vsum, GY, '-',  1.3);
    hHw = q(Vhw,  GH, '--', 1.6);
    hRw = q(Vrw,  BL, '--', 1.6);
    hSw = q(Vs,   RD, '--', 1.8);

    legend([hHr hRr hSr hHw hRw hSw], ...
           {'v_h','v_r','v_h + v_r', ...
            '\eta_h v_h','\eta_r v_r','v_s (command)'}, ...
           'Location','eastoutside');
    xlabel(ax,'in-plane u (m)');  ylabel(ax,'in-plane w (m)');
    [~, nm, ex] = fileparts(csvfile);
    ttl = sprintf('%s   |   %s   |   arrows every %.1f s  (scale \\times%.1f)', ...
        strrep(char(string(T.condition(1))),'_','\_'), ...
        strrep([nm ex],'_','\_'), arrow_dt, g);
    if ~isempty(lap_only), ttl = sprintf('%s   |   lap %d', ttl, lap_only); end
    title(ax, ttl);

    fprintf('  paths: geometry from %s;  %d arrow times, scale x%.1f\n', ...
            geom_src, numel(idx), g);
end

% ------------------------------------------------------------------
function [C, R, N, src] = load_sidecar_geom(csvfile)
    C = [0.45 0 0.45];  R = 0.05;  N = [0 0 1];
    src = 'defaults [0.45 0 0.45] r=0.05 n=[0 0 1]';
    [d, n] = fileparts(csvfile);
    side = fullfile(d, [n '.params.json']);
    if exist(side, 'file') ~= 2, return; end
    try
        s = jsondecode(fileread(side));
        C = reshape(s.path.center, 1, []);
        R = double(s.path.radius);
        N = reshape(s.path.normal, 1, []);
        src = [n '.params.json'];
    catch ME
        warning('sidecar %s geometry unreadable (%s); using defaults', ...
                side, ME.message);
    end
end
