function source_leakage_combined(downloads, reference_path, output)
% Rebuild only the illustrative 104012/LH source-resolution matrix (Figure S1).
% The stored run-10 orientation and depth scale define common source units.
ftpath = getenv('FIELDTRIP_HOME');
if isempty(ftpath), ftpath = fullfile(fileparts(mfilename('fullpath')), 'fieldtrip-20251103'); end
addpath(ftpath); ft_defaults;
ref = load(reference_path);
base = fullfile(downloads, '104012', 'MEG');
load(fullfile(base, 'anatomy', '104012_MEG_anatomy_headmodel.mat'), 'headmodel');
load(fullfile(base, 'anatomy', '104012_MEG_anatomy_sourcemodel_2d.mat'), 'sourcemodel2d');
headmodel = ft_convert_units(headmodel, 'cm');
sourcemodel2d = ft_convert_units(sourcemodel2d, 'cm');
R = zeros(numel(ref.inside_indices));
counts = zeros(1,2);
for j = 1:2
    run = 9+j;
    load(fullfile(base,'Motort','tmegpreproc',sprintf('104012_MEG_%d-Motort_tmegpreproc_TEMG.mat',run)), 'data');
    data.grad = ft_convert_units(data.grad,'cm');
    selected = find(data.trialinfo(:,2)==1);
    cfg = []; cfg.channel = {'MEG'}; cfg.trials = selected;
    data = ft_selectdata(cfg,data);
    cfg = []; cfg.demean = 'yes'; cfg.detrend = 'yes';
    cfg.bpfilter = 'yes'; cfg.bpfreq = [0.5,100];
    clean = ft_preprocessing(cfg,data); clear data;
    cfg = []; cfg.covariance = 'yes'; cfg.covariancewindow = 'all'; cfg.keeptrials = 'no';
    tl = ft_timelockanalysis(cfg,clean);
    cfg = []; cfg.grad = clean.grad; cfg.headmodel = headmodel;
    cfg.sourcemodel = sourcemodel2d; cfg.channel = tl.label;
    cfg.reducerank = 2; cfg.normalize = 'no';
    lf = ft_prepare_leadfield(cfg,tl);
    inside = find(lf.inside);
    assert(isequal(inside(:),double(ref.inside_indices(:))), 'Source grid mismatch');
    for k = 1:numel(inside)
        idx = inside(k);
        lf.leadfield{idx} = (lf.leadfield{idx}/ref.leadfield_scale(k))*ref.source_orientation(:,k);
    end
    cfg = []; cfg.method = 'lcmv'; cfg.sourcemodel = lf;
    cfg.headmodel = headmodel; cfg.keepleadfield = 'yes';
    cfg.lcmv.keepfilter = 'yes'; cfg.lcmv.projectnoise = 'yes';
    cfg.lcmv.weightnorm = 'unitgain'; cfg.lcmv.lambda = '5%';
    cfg.lcmv.reducerank = 2; cfg.lcmv.fixedori = 'no';
    src = ft_sourceanalysis(cfg,tl);
    W = vertcat(src.avg.filter{inside});
    L = horzcat(lf.leadfield{inside});
    assert(max(abs(sum(W.*L',2)-1)) < 1e-5, 'Unit-gain constraint failed');
    % Confirm that the rebuilt operators agree with the frozen parcel inputs.
    expectedW = ref.(sprintf('W%d',run)); expectedL = ref.(sprintf('L%d',run));
    assert(norm(ref.P*W-expectedW,'fro')/norm(expectedW,'fro') < 1e-6, 'Parcel filter mismatch');
    assert(norm(L*ref.U-expectedL,'fro')/norm(expectedL,'fro') < 1e-10, 'Parcel leadfield mismatch');
    counts(j) = numel(clean.trial);
    R = R + counts(j)*(W*L);
end
assert(isequal(counts,ref.counts), 'Trial count mismatch');
R = R/sum(counts);
assert(norm(ref.P*R*ref.U-ref.resolution,'fro')/norm(ref.resolution,'fro') < 1e-6, 'Combined resolution mismatch');
check.pos = sourcemodel2d.pos(inside,:);
runs = [10,11]; subject = '104012'; condition = 'LH';
save(output,'R','check','runs','counts','subject','condition','-v7.3');
fprintf('Combined source-resolution matrix saved: %s\n',output);
end
