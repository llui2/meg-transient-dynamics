function beamform_runs(subject, condition, output, atlas_path, downloads, rois)
% Reconstruct both motor runs in the same signed source coordinates.
% Run 10 determines the orientation; each run has its own sensors, covariance,
% leadfields and unit-gain filters. No model fitting is performed here.
ftpath = getenv('FIELDTRIP_HOME');
if isempty(ftpath), ftpath = fullfile(fileparts(mfilename('fullpath')), 'fieldtrip-20251103'); end
addpath(ftpath); ft_defaults;
base = fullfile(downloads, subject, 'MEG');
load(fullfile(base, 'anatomy', [subject '_MEG_anatomy_headmodel.mat']), 'headmodel');
load(fullfile(base, 'anatomy', [subject '_MEG_anatomy_sourcemodel_2d.mat']), 'sourcemodel2d');
headmodel = ft_convert_units(headmodel, 'cm');
sourcemodel2d = ft_convert_units(sourcemodel2d, 'cm');
atlas = load(atlas_path);
names = {'LH', 'LF', 'RH', 'RF'}; codes = [1, 2, 4, 5];
code = codes(strcmp(names, condition));
assert(isscalar(code), 'Unknown movement condition');
if ~exist(output, 'dir'), mkdir(output); end
orientation = []; reference_inside = []; leadfield_scale = [];
for run = [10, 11]
    input_path = fullfile(base, 'Motort', 'tmegpreproc', ...
        sprintf('%s_MEG_%d-Motort_tmegpreproc_TEMG.mat', subject, run));
    load(input_path, 'data');
    data.grad = ft_convert_units(data.grad, 'cm');
    trial_indices = find(data.trialinfo(:,2) == code);
    assert(~isempty(trial_indices), 'No selected trials');
    trialinfo = data.trialinfo(trial_indices,:);
    cfg = []; cfg.channel = {'MEG'}; cfg.trials = trial_indices;
    data = ft_selectdata(cfg, data);
    cfg = []; cfg.demean = 'yes'; cfg.detrend = 'yes';
    cfg.bpfilter = 'yes'; cfg.bpfreq = [0.5, 100];
    data_clean = ft_preprocessing(cfg, data); clear data;
    cfg = []; cfg.covariance = 'yes'; cfg.covariancewindow = 'all';
    cfg.keeptrials = 'no';
    tl = ft_timelockanalysis(cfg, data_clean);
    cfg = []; cfg.grad = data_clean.grad; cfg.headmodel = headmodel;
    cfg.sourcemodel = sourcemodel2d; cfg.channel = tl.label;
    cfg.reducerank = 2; cfg.normalize = 'no';
    lf = ft_prepare_leadfield(cfg, tl);
    % Match the original normalize='yes', normalizeparam=1 for run 10.
    % Reuse those factors for run 11: independent depth normalization would
    % otherwise give the same physical dipole different units in each run.
    if run == 10
        leadfield_scale = zeros(numel(lf.inside), 1);
        for idx = find(lf.inside(:))'
            leadfield_scale(idx) = sum(lf.leadfield{idx}(:).^2);
        end
    else
        assert(isequal(find(lf.inside), reference_inside), 'Leadfield source grid differs');
    end
    for idx = find(lf.inside(:))'
        assert(leadfield_scale(idx) > 0, 'Zero reference leadfield');
        lf.leadfield{idx} = lf.leadfield{idx} / leadfield_scale(idx);
    end
    cfg = []; cfg.method = 'lcmv'; cfg.sourcemodel = lf;
    cfg.headmodel = headmodel; cfg.keepleadfield = 'yes';
    cfg.lcmv.keepfilter = 'yes'; cfg.lcmv.projectnoise = 'yes';
    cfg.lcmv.weightnorm = 'unitgain'; cfg.lcmv.lambda = '5%';
    cfg.lcmv.reducerank = 2;
    if run == 10
        cfg.lcmv.fixedori = 'yes';
    else
        % Project explicitly: ft_sourceanalysis strips a supplied mom field
        % when accepting precomputed leadfields in this FieldTrip release.
        % Re-optimizing orientation would make signed trial pooling ambiguous.
        for idx = reference_inside(:)'
            cfg.sourcemodel.leadfield{idx} = lf.leadfield{idx} * orientation(:,idx);
        end
        cfg.lcmv.fixedori = 'no';
    end
    src = ft_sourceanalysis(cfg, tl);
    inside_idx = find(src.inside);
    if run == 10
        reference_inside = inside_idx;
        orientation = zeros(3, numel(src.inside));
        orientation(:,inside_idx) = horzcat(src.avg.ori{inside_idx});
    else
        assert(isequal(inside_idx, reference_inside), 'Source grid differs between runs');
    end
    W = vertcat(src.avg.filter{inside_idx});
    assert(size(W,1) == numel(inside_idx), 'Expected scalar source filters');
    assert(isequal(tl.label, data_clean.label), 'Sensor channel order mismatch');
    assert(isequal(lf.label, tl.label), 'Leadfield channel order mismatch');
    L = zeros(size(W,2), numel(inside_idx));
    for k = 1:numel(inside_idx)
        idx = inside_idx(k);
        L(:,k) = lf.leadfield{idx} * orientation(:,idx);
    end
    unit_gain_error = max(abs(sum(W .* L', 2) - 1));
    assert(unit_gain_error < 1e-5, 'Unit-gain constraint failed');
    channel_labels = data_clean.label;
    source_positions = sourcemodel2d.pos(inside_idx,:);
    source_orientation = orientation(:,inside_idx);
    time = double(data_clean.time{1}(:)'); fs = double(data_clean.fsample);
    n_trials = numel(data_clean.trial);
    for k = 1:n_trials
        assert(isequal(data_clean.time{k}(:)', time), 'Trial time grids differ');
        assert(all(isfinite(data_clean.trial{k}(:))), 'Nonfinite sensor input');
    end
    clear src lf tl;
    for roi = rois
        full_labels = double(atlas.(sprintf('labels%d', roi)));
        assert(numel(full_labels) == numel(orientation(1,:)), 'Atlas/source grid mismatch');
        labels_inside = full_labels(inside_idx);
        parcel_labels = unique(labels_inside(labels_inside > 0));
        assert(isequal(parcel_labels(:), (1:roi)'), 'Missing parcel labels');
        P = zeros(roi, numel(inside_idx)); U = P';
        for p = 1:roi
            indices = find(labels_inside == parcel_labels(p));
            P(p,indices) = 1 / numel(indices); U(indices,p) = 1;
        end
        W_roi = P * W; L_roi = L * U; R_roi = W_roi * L_roi;
        % Project directly to parcels: source trajectories need not occupy disk.
        X_trials = zeros(roi, numel(time), n_trials, 'single');
        for k = 1:n_trials
            X_trials(:,:,k) = single(W_roi * double(data_clean.trial{k}));
        end
        X_avg = mean(double(X_trials), 3);
        save(fullfile(output, sprintf('run%d_parcel_%d.mat', run, roi)), ...
            'X_trials', 'X_avg', 'R_roi', 'W_roi', 'L_roi', 'parcel_labels', ...
            'time', 'fs', 'n_trials', 'trial_indices', 'trialinfo', ...
            'channel_labels', 'source_positions', 'source_orientation', ...
            'inside_idx', 'leadfield_scale', 'unit_gain_error', 'input_path', 'run', '-v7');
        clear X_trials X_avg P U W_roi L_roi R_roi;
    end
    fprintf('Reconstructed %s %s run %d: %d trials, %d parcels sets.\n', ...
        subject, condition, run, n_trials, numel(rois));
    clear data_clean W L;
end
end
