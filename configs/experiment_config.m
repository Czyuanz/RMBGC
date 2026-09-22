function cfg = experiment_config(mode)
%EXPERIMENT_CONFIG Search and evaluation configuration used in the paper.
if nargin < 1, mode = 'formal'; end
cfg = struct();
cfg.tau = [0.1 0.2 0.3];
cfg.delta_grid = [0.10 0.15 0.30 0.50];
cfg.lambda_grid = [0.005 0.01 0.03];
cfg.tuning_seeds = 1:10;
cfg.evaluation_seeds = 11:30;
cfg.max_iter = 50;
cfg.tolerance = 1e-6;
cfg.n_neighbors = 8;
cfg.anchor_seed = 20260803;
cfg.anchor_max_iter = 100;
cfg.anchor_replicates = 3;
cfg.selection_metric = 'mean((ARI+MacroF1)/2)';
if strcmpi(mode,'smoke')
    cfg.delta_grid = cfg.delta_grid(1:2);
    cfg.lambda_grid = cfg.lambda_grid(1:2);
    cfg.tuning_seeds = cfg.tuning_seeds(1:2);
    cfg.evaluation_seeds = cfg.evaluation_seeds(1:2);
    cfg.tau = cfg.tau(1);
elseif ~strcmpi(mode,'formal')
    error('experiment_config:Mode','Mode must be smoke or formal.');
end
end
