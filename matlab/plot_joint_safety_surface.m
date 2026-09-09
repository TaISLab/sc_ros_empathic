function plot_joint_safety_surface(limits_deg, opts)
%PLOT_JOINT_SAFETY_SURFACE  eta_k3 (the joint-limit-safety factor) as a
%   height surface over (joint angle, joint velocity) -- a method figure.
%
%   X  = joint angle q_i (deg), with the limits and the caution
%        (proximity_threshold) values marked on the floor.
%   Y  = joint velocity qdot_i (rad/s), negative .. positive.
%   Z  = eta_k3 in (0, 1].
%
%   plot_joint_safety_surface()                 % generic joint, q in [-90, 90]
%   plot_joint_safety_surface([0 145])          % e.g. the elbow, deg
%   plot_joint_safety_surface([qmin qmax], opts)
%
%   opts (struct; defaults = config/shared_control.yaml):
%     .Cs                 12
%     .proximity_threshold 0.3       (tau; caution band = rho in [-(1-tau), (1-tau)])
%     .margin_floor        0.05
%     .static_weight       0.5
%     .dynamic_weight      0.5
%     .qdot_max            1.5        rad/s, Y half-range
%     .n                   161        grid points per axis
%     .style              'surf'      | 'contourf'
%
%   This mirrors performance.joint_safety_factor for ONE joint in
%   isolation (the other three assumed mid-range -> zero contribution),
%   so it shows the SHAPE of the factor, not a trial. In the controller
%   qdot_i is the joint velocity the candidate Cartesian command induces
%   through the human-arm Jacobian; here it is swept directly.
%
%   closing_rate = qdot_i*sign(rho_i)   (>0 approach, <0 retreat)
%   penalty = max(0,  static_weight  * tau * prox_w
%                   + dynamic_weight * prox_w * closing_rate
%                                      / max(margin_i, margin_floor) )
%   with margin_i = 1 - |rho_i|,  prox_w = clip((tau - margin_i)/tau, 0, 1),
%   rho_i = (q_i - q_mid)/q_half,  eta_k3 = exp(-Cs * penalty).
%   The dynamic term is SIGNED: a retreat earns a relief credit that
%   offsets the static term (per joint, the sum is floored at 0).

    if nargin < 1 || isempty(limits_deg), limits_deg = [-90 90]; end
    d.Cs = 12; d.proximity_threshold = 0.3; d.margin_floor = 0.05;
    d.static_weight = 0.5; d.dynamic_weight = 0.5;
    d.qdot_max = 1.5; d.n = 161; d.style = 'surf';
    if nargin < 2, opts = struct(); end
    for f = fieldnames(d).'
        if ~isfield(opts, f{1}) || isempty(opts.(f{1}))
            opts.(f{1}) = d.(f{1});
        end
    end

    qmin = limits_deg(1); qmax = limits_deg(2);
    qmid = 0.5 * (qmin + qmax);  qhalf = 0.5 * (qmax - qmin);
    tau = opts.proximity_threshold;

    q  = linspace(qmin, qmax, opts.n);                 % deg
    qd = linspace(-opts.qdot_max, opts.qdot_max, opts.n);   % rad/s
    [Q, QD] = meshgrid(q, qd);

    RHO    = (Q - qmid) ./ qhalf;
    MARGIN = 1 - abs(RHO);
    PROXW  = min(max((tau - MARGIN) ./ tau, 0), 1);
    CLOSE  = QD .* sign(RHO);                          % >0 approach, <0 retreat
    STAT   = tau .* PROXW;
    DYN    = PROXW .* CLOSE ./ max(MARGIN, opts.margin_floor);   % signed
    PEN    = max(0, opts.static_weight .* STAT + opts.dynamic_weight .* DYN);
    ETA    = exp(-opts.Cs .* PEN);

    qc = qmid + (1 - tau) * qhalf * [-1 1];            % caution q (rho = +-(1-tau))
    ym = opts.qdot_max;

    figure('Color', 'w', 'Position', [100 100 760 620]);
    ax = axes; hold(ax, 'on');

    if strcmpi(opts.style, 'contourf')
        contourf(ax, Q, QD, ETA, 0:0.05:1, 'LineColor', 'none');
        for xq = [qmin qmax]
            plot(ax, [xq xq], [-ym ym], 'r--', 'LineWidth', 1.6);
        end
        for xq = qc
            plot(ax, [xq xq], [-ym ym], ':', 'Color', [.85 .54 0], ...
                 'LineWidth', 1.6);
        end
        plot(ax, [qmid qmid], [-ym ym], '-', 'Color', [.5 .5 .5], ...
             'LineWidth', 0.8);
        view(ax, 2);
    else
        surf(ax, Q, QD, ETA, 'EdgeColor', 'none');
        shading(ax, 'interp');
        contour3(ax, Q, QD, ETA, 0.1:0.2:0.9, 'Color', [.3 .3 .3]);
        zlim(ax, [0 1]);  zlabel(ax, '\eta_{k3}');
        % limit / caution / mid-range as vertical curtains on the floor
        for xq = [qmin qmax]
            plot3(ax, [xq xq], [-ym ym], [0 0], 'r--', 'LineWidth', 1.8);
        end
        for xq = qc
            plot3(ax, [xq xq], [-ym ym], [0 0], ':', 'Color', [.85 .54 0], ...
                  'LineWidth', 1.8);
        end
        plot3(ax, [qmid qmid], [-ym ym], [0 0], '-', 'Color', [.5 .5 .5], ...
              'LineWidth', 0.8);
        view(ax, [-38 28]);
    end

    caxis(ax, [0 1]);
    colormap(ax, parula);
    cb = colorbar(ax);  cb.Label.String = '\eta_{k3}';
    grid(ax, 'on');  box(ax, 'on');
    xlabel(ax, 'joint angle  q_i  (deg)');
    ylabel(ax, 'joint velocity  $\dot q_i$  (rad/s)', 'Interpreter', 'latex');
    xlim(ax, [qmin qmax]);  ylim(ax, [-ym ym]);
    title(ax, sprintf(['joint-limit-safety factor  \\eta_{k3}   ', ...
        '(C_s = %g,  \\tau = %g)'], opts.Cs, tau), 'Interpreter', 'tex');

    h = [plot(ax, nan, nan, 'r--', 'LineWidth', 1.8), ...
         plot(ax, nan, nan, ':', 'Color', [.85 .54 0], 'LineWidth', 1.8), ...
         plot(ax, nan, nan, '-', 'Color', [.5 .5 .5], 'LineWidth', 0.8)];
    legend(h, {'joint limit', ...
               sprintf('caution (\\rho = \\pm%.2g)', 1 - tau), ...
               'mid-range'}, 'Location', 'northoutside', ...
               'Orientation', 'horizontal', 'Box', 'off');

    fprintf(['plot_joint_safety_surface: q in [%.0f, %.0f] deg, ', ...
             'caution at %.1f / %.1f deg, qdot in [%.2f, %.2f] rad/s\n'], ...
             qmin, qmax, qc(1), qc(2), -ym, ym);
end
