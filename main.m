% =========================================================================
% MEG source reconstruction for multiple HCP subjects using LCMV
% =========================================================================

clear; clc; close all;
restoredefaultpath;
addpath('fieldtrip-20251103');
evalc('ft_defaults');

% -------------------------------------------------------------------------
% CONFIGURATION
% -------------------------------------------------------------------------

subj_list = {'104012','105923','106521','108323','109123','113922','116726','133019','140117','156334','162026','162935','164636','169040','175237','177746','185442','191033','191437','192641','198653','204521','205119','212318','212823'};

%subj_list = {'255639','257845','283543','287248','293748','353740','358144','406836','559053','568963','581450','599671','601127','660951','662551','667056','679770','680957','706040','707749','725751','735148','783462','814649','891667','898176'};

sess  = '10';
task  = 'Motort';
lock  = 'TEMG';
conds = {'LH'};

root_dir = '/home/lluis/Downloads';

cond_map = struct('LH',1,'LF',2,'RH',4,'RF',5,'Fix',6);
cond_str = strjoin(conds, '');

% =========================================================================
% MAIN LOOP
% =========================================================================
for i = 1:numel(subj_list)
    subj = subj_list{i};

    fprintf('\n============================================================\n');
    fprintf('[1/9] Subject %s (%d/%d)\n', subj, i, numel(subj_list));
    fprintf('============================================================\n');

    base_dir = fullfile(root_dir, subj, 'MEG', task, 'tmegpreproc');
    out_dir  = fullfile(base_dir, sprintf('source_trials_%s_%s', lock, cond_str));
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    % ---------------------------------------------------------------------
    % STEP 1: LOAD DATA
    % ---------------------------------------------------------------------
    fprintf('[2/9] Loading data\n');

    data_file = fullfile(base_dir, ...
        sprintf('%s_MEG_%s-%s_tmegpreproc_%s.mat', subj, sess, task, lock));

    if ~exist(data_file, 'file')
        warning('Missing data for %s. Skipping.', subj);
        continue;
    end

    try
        load(data_file, 'data');
        load(fullfile(root_dir, subj, 'MEG', 'anatomy', ...
            [subj '_MEG_anatomy_headmodel.mat']), 'headmodel');
        load(fullfile(root_dir, subj, 'MEG', 'anatomy', ...
            [subj '_MEG_anatomy_sourcemodel_2d.mat']), 'sourcemodel2d');
    catch ME
        warning('Load failed for %s: %s', subj, ME.message);
        continue;
    end

    % Enforce consistent spatial units
    data.grad     = ft_convert_units(data.grad, 'cm');
    headmodel     = ft_convert_units(headmodel, 'cm');
    sourcemodel2d = ft_convert_units(sourcemodel2d, 'cm');

    % ---------------------------------------------------------------------
    % STEP 2: TRIAL SELECTION
    % ---------------------------------------------------------------------
    fprintf('[3/9] Selecting trials\n');

    use_codes = cellfun(@(c) cond_map.(upper(c)), conds);
    trial_codes = data.trialinfo(:,2);
    selected_trials = find(ismember(trial_codes, use_codes));

    if isempty(selected_trials)
        warning('No matching trials for %s. Skipping.', subj);
        continue;
    end

    cfg = [];
    cfg.channel = {'MEG'};
    cfg.trials  = selected_trials;
    evalc('data = ft_selectdata(cfg, data)');

    % ---------------------------------------------------------------------
    % STEP 3: PREPROCESSING AND COVARIANCE
    % ---------------------------------------------------------------------
    fprintf('[4/9] Preprocessing\n');

    cfg = [];
    cfg.demean   = 'yes';
    cfg.detrend  = 'yes';
    cfg.bpfilter = 'yes';
    cfg.bpfreq   = [0.5 100];
    evalc('data_clean = ft_preprocessing(cfg, data)');

    cfg = [];
    cfg.covariance       = 'yes';
    cfg.covariancewindow = 'all';
    cfg.keeptrials       = 'no';
    evalc('tl = ft_timelockanalysis(cfg, data_clean)');

    % ---------------------------------------------------------------------
    % STEP 4: LEADFIELD
    % ---------------------------------------------------------------------
    fprintf('[5/9] Leadfield\n');

    cfg = [];
    cfg.grad        = data_clean.grad;
    cfg.headmodel   = headmodel;
    cfg.sourcemodel = sourcemodel2d;
    cfg.channel     = tl.label;
    cfg.reducerank  = 2;
    cfg.normalize   = 'yes';
    cfg.normalizeparam = 1;

    evalc('lf = ft_prepare_leadfield(cfg, tl)');

    % ---------------------------------------------------------------------
    % STEP 5: LCMV BEAMFORMER
    % ---------------------------------------------------------------------
    fprintf('[6/9] LCMV beamformer\n');

    cfg = [];
    cfg.method            = 'lcmv';
    cfg.sourcemodel       = lf;
    cfg.headmodel         = headmodel;
    cfg.keepleadfield     = 'yes';
    cfg.lcmv.keepfilter   = 'yes';
    cfg.lcmv.fixedori     = 'yes';
    cfg.lcmv.projectnoise = 'yes';
    cfg.lcmv.weightnorm   = 'unitgain';
    cfg.lcmv.reducerank   = 2;
    cfg.lcmv.lambda       = '5%';

    evalc('src = ft_sourceanalysis(cfg, tl)');

    % ---------------------------------------------------------------------
    % STEP 6: RESOLUTION MATRIX
    % ---------------------------------------------------------------------
    fprintf('[7/9] Resolution matrix\n');

    inside_idx = find(src.inside);
    n_sources  = numel(inside_idx);

    W = vertcat(src.avg.filter{inside_idx});
    L = zeros(size(W, 2), n_sources);

    for k = 1:n_sources
        idx = inside_idx(k);
        L(:,k) = lf.leadfield{idx} * src.avg.ori{idx};
    end

    R = W * L;

    % ---------------------------------------------------------------------
    % STEP 7: SAVE METADATA
    % ---------------------------------------------------------------------
    fprintf('[8/9] Saving metadata\n');

    meta.fs     = data_clean.fsample;
    meta.inside = src.inside;
    meta.time   = data_clean.time{1};
    meta.R      = R;
    meta.check.pos    = single(sourcemodel2d.pos(src.inside,:));
    meta.check.labels = uint32(inside_idx);

    save(fullfile(out_dir,'meta.mat'),'-struct','meta','-v7.3');

    % ---------------------------------------------------------------------
    % STEP 8: PROJECT TRIALS
    % ---------------------------------------------------------------------
    fprintf('[9/9] Projecting trials\n');

    for k = 1:numel(data_clean.trial)
        SourceData = W * single(data_clean.trial{k});
        save(fullfile(out_dir, sprintf('source_trial_%03d.mat',k)), 'SourceData');
    end

    fprintf('DONE with %s\n', subj);
end

fprintf('\nAll %d subjects processed.\n', numel(subj_list));
