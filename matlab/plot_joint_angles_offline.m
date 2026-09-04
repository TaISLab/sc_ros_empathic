function plot_joint_angles_offline(csvfile, window, limits_deg)
%PLOT_JOINT_ANGLES_OFFLINE  Offline view of an sc_ros_empathic trial CSV,
%   mirroring the live plot_joint_angles.py: one subplot per human joint
%   (q1..q4 in degrees) with its [min,max] band and the proximity-threshold
%   lines, plus a joint_safety / eta subplot. Shared, zoomable time axis.
%
%   plot_joint_angles_offline(csvfile)
%   plot_joint_angles_offline(csvfile, [t0 t1])           % seconds from start
%   plot_joint_angles_offline(csvfile, N)                 % just lap N (scalar)
%   plot_joint_angles_offline(csvfile, sel, limits)       % + 4x2 limits, deg
%
%   The CSV logs rho1..rho4 (signed position in [-1,1], 0 = mid-range,
%   +-1 = a limit) and m1..m4, not the raw angles. q_i is reconstructed as
%   qmid_i + rho_i*qhalf_i from LIMITS. If a "<csv>.params.json" sidecar
%   (written by shared_control_node) sits next to the CSV, its
%   human_model.joint_limits_rad are used automatically; otherwise the
%   study DEFAULT_JOINT_LIMITS -- what a trial run WITHOUT a subject file
%   uses: q1 [-60,180] q2 [0,180] q3 [-90,90] q4 [0,145] (deg). An
%   explicit LIMITS argument overrides both.
%
%   Zoom/pan any subplot -- the time axes are linked. A one-line summary
%   (duration, laps, median eta, time spent past a limit) is printed too.
%
%   Requires MATLAB R2019b+ (tiledlayout, xline).

    DEFAULT_LIMITS = [-60 180; 0 180; -90 90; 0 145];   % deg
    if nargin < 2, window = []; end
    TAU = 0.3;                        % proximity_threshold (config default)
    JC = [ 31 119 180; 214 39 40; 44 160 44; 148 103 189] / 255;
    JN = {'q1 shoulder flex/ext','q2 shoulder abd/add', ...
          'q3 shoulder int/ext rot','q4 elbow (0=extended)'};

    % joint limits: explicit 3rd arg > <csv>.params.json sidecar > defaults
    if nargin >= 3 && ~isempty(limits_deg)
        limits_src = 'argument';
    else
        [limits_deg, limits_src] = load_sidecar_limits(csvfile, DEFAULT_LIMITS);
    end

    T = readtable(csvfile);
    t = T.t_rel - T.t_rel(1);        % seconds from the first logged row

    % ---- row selection --------------------------------------------
    lap_only = [];
    if isempty(window)
        sel = true(height(T),1);
    elseif isscalar(window)                     % a lap number
        lap_only = window;
        sel = T.lap == lap_only;
    else                                        % [t0 t1] seconds
        sel = t >= window(1) & t <= window(2);
    end
    assert(any(sel), 'No rows in the requested window / lap.');

    % ---- reconstruct q_i (deg) ------------------------------------
    rho   = [T.rho1 T.rho2 T.rho3 T.rho4];
    qmid  = mean(limits_deg, 2).';
    qhalf = 0.5 * diff(limits_deg, 1, 2).';
    q     = qmid + rho .* qhalf;                % height(T) x 4, deg

    tf = t(sel);  qf = q(sel,:);  rf = rho(sel,:);
    js = T.joint_safety_h(sel);
    eh = T.eta_h(sel);  es = T.eta_s(sel);
    fresh = T.human_fresh(sel) == 1;
    lap = T.lap(sel);
    xr = [min(tf) max(tf)];

    lapEdges = tf([true; diff(lap) ~= 0]);
    lapNums  = lap([true; diff(lap) ~= 0]);

    % ---- figure --------------------------------------------------
    figure('Name','joint angles (offline)','Color','w', ...
           'Position',[80 60 1000 900]);
    tl = tiledlayout(5,1,'TileSpacing','compact','Padding','compact');
    ax = gobjects(5,1);

    for i = 1:4
        ax(i) = nexttile; hold(ax(i),'on');
        lo = limits_deg(i,1); hi = limits_deg(i,2);
        pth = qmid(i) + (1-TAU)*qhalf(i)*[-1 1];        % rho = +-0.7
        shade_stale(ax(i), tf, fresh);                  % behind everything
        fill(ax(i), [xr fliplr(xr)], [lo lo hi hi], JC(i,:), ...
             'FaceAlpha',0.06, 'EdgeColor','none', 'HandleVisibility','off');
        plot(ax(i), xr, [lo lo], '--', 'Color',JC(i,:), 'LineWidth',1);
        plot(ax(i), xr, [hi hi], '--', 'Color',JC(i,:), 'LineWidth',1);
        plot(ax(i), xr, [1 1]*pth(1), ':', 'Color',[.85 .55 0], 'LineWidth',1);
        plot(ax(i), xr, [1 1]*pth(2), ':', 'Color',[.85 .55 0], 'LineWidth',1);
        plot(ax(i), tf, qf(:,i), '-', 'Color',JC(i,:), 'LineWidth',1.4);
        qv = qf(~isnan(qf(:,i)), i); if isempty(qv), qv = [lo; hi]; end
        pad = 0.05*(hi-lo);
        ylim(ax(i), [min(lo, min(qv))-pad, max(hi, max(qv))+pad]);
        ylabel(ax(i), sprintf('%s\n(deg)', JN{i}));
        grid(ax(i),'on');
        draw_lap_lines(ax(i), lapEdges, lapNums, i==1);
    end

    ax(5) = nexttile; hold(ax(5),'on');
    shade_stale(ax(5), tf, fresh);
    plot(ax(5), tf, js, '-',  'Color',[.85 .10 .10], 'LineWidth',1.6);
    plot(ax(5), tf, eh, '--', 'Color',[.10 .10 .10], 'LineWidth',1.0);
    plot(ax(5), tf, es, '-.', 'Color',[.45 .45 .45], 'LineWidth',1.0);
    ylim(ax(5), [-0.02 1.05]); grid(ax(5),'on');
    ylabel(ax(5), 'joint\_safety_h / eta');
    legend(ax(5), {'joint\_safety_h','eta_h','eta_s'}, ...
           'Location','southoutside','Orientation','horizontal','Box','off');
    draw_lap_lines(ax(5), lapEdges, lapNums, false);
    xlabel(ax(5), 't (s)');

    linkaxes(ax, 'x');
    xlim(ax(5), xr);
    [~, nm, ex] = fileparts(csvfile);
    ttl = sprintf('%s   |   %s', strrep(char(string(T.condition(1))),'_','\_'), ...
                  strrep([nm ex],'_','\_'));
    if ~isempty(lap_only), ttl = sprintf('%s   |   lap %d', ttl, lap_only); end
    title(tl, ttl);

    % ---- console summary --------------------------------------
    snf = T.s_near(sel);
    wraps = nnz(snf(1:end-1) > 0.8 & snf(2:end) < 0.2);
    fprintf('\n%s\n', csvfile);
    fprintf('  joint limits: %s\n', limits_src);
    fprintf('  window %.1f-%.1f s  (%.1f s, ~%d laps)\n', ...
            xr(1), xr(2), xr(2)-xr(1), wraps);
    fprintf('  eta_h med %.3f   eta_s med %.3f\n', ...
            median(eh,'omitnan'), median(es,'omitnan'));
    fprintf('  joint_safety_h med %.3f   < 0.05 for %.0f%% of the window\n', ...
            median(js,'omitnan'), 100*mean(js(~isnan(js)) < 0.05));
    for i = 1:4
        ri = rf(~isnan(rf(:,i)), i);
        fprintf('  q%d: rho med %+.2f   past a limit %.0f%%   inside tau %.0f%%\n', ...
                i, median(ri), 100*mean(abs(ri) > 1), 100*mean(abs(ri) > 1-TAU));
    end

    % ---- companion figure: traced paths + velocity arrows ---------
    try
        plot_paths_offline(csvfile, window);
    catch ME
        warning('plot_paths_offline failed: %s', ME.message);
    end
end

% -------------------------------------------------------------------
function shade_stale(ax, t, fresh)
% grey vertical bands where human_fresh == 0 (joint data stale/NaN there)
    stale = ~fresh(:);
    if ~any(stale), return; end
    d = diff([0; stale; 0]);
    s = find(d == 1);  e = find(d == -1) - 1;
    for k = 1:numel(s)
        a = t(s(k)); b = t(e(k));
        if b > a
            fill(ax, [a b b a], [-1e4 -1e4 1e4 1e4], [.6 .6 .6], ...
                 'FaceAlpha',0.12, 'EdgeColor','none', 'HandleVisibility','off');
        end
    end
end

function [lim_deg, src] = load_sidecar_limits(csvfile, default_deg)
% Use human_model.joint_limits_rad from "<csv>.params.json" if present.
    lim_deg = default_deg;  src = 'DEFAULT_JOINT_LIMITS';
    [d, n] = fileparts(csvfile);
    side = fullfile(d, [n '.params.json']);
    if exist(side, 'file') ~= 2, return; end
    try
        s = jsondecode(fileread(side));
        jl = s.human_model.joint_limits_rad;      % 4x2 rad (or flat 8)
        if isvector(jl), jl = reshape(jl(:), 2, 4).'; end
        assert(isequal(size(jl), [4 2]), 'joint_limits_rad not 4x2');
        lim_deg = jl * 180/pi;
        src = sprintf('%s sidecar (%s)', [n '.params.json'], ...
                      getfield_default(s.human_model, 'joint_limits_source', '?'));
    catch ME
        warning('sidecar %s unreadable (%s); using defaults', side, ME.message);
    end
end

function v = getfield_default(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end

function draw_lap_lines(ax, edges, nums, label)
    for k = 1:numel(edges)
        xline(ax, edges(k), '-', 'Color',[.7 .7 .7], 'HandleVisibility','off');
        if label
            text(ax, edges(k), max(ax.YLim), sprintf(' L%d', nums(k)), ...
                 'VerticalAlignment','top', 'FontSize',7, 'Color',[.4 .4 .4]);
        end
    end
end
