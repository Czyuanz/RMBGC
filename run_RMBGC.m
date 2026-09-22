function output = run_RMBGC(dataFile,mode)
%RUN_RMBGC Tune and evaluate RMBGC under the paper's protocol.
%   OUTPUT = RUN_RMBGC(DATAFILE,'formal') loads a MAT file containing Xs
%   (or X) and Y (or label), searches delta and lambda on seeds 1--10,
%   evaluates the selected pair on seeds 11--30 at tau = 0.1, 0.2, 0.3,
%   and writes CSV files under results/<dataset>/.
if nargin < 2, mode = 'formal'; end
root = fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'utils'),fullfile(root,'measures'),fullfile(root,'configs'));
cfg = experiment_config(mode);
[Xs,Y,dataset] = load_public_dataset(dataFile);
c = numel(unique(Y));
[B,graphInfo] = build_real_bipartite_graphs(Xs,c,'NAnchors',0, ...
    'NNeighbors',cfg.n_neighbors,'AnchorSeed',cfg.anchor_seed, ...
    'AnchorMaxIter',cfg.anchor_max_iter,'AnchorReplicates',cfg.anchor_replicates, ...
    'AnchorMethod','kmeans');

outDir = fullfile(root,'results',dataset);
if ~isfolder(outDir), mkdir(outDir); end
tuning = run_tuning(B,Y,c,cfg);
writetable(tuning,fullfile(outDir,'tuning_per_seed.csv'));
summary = groupsummary(tuning,{'Delta','Lambda'},'mean',{'ARI','MacroF1'});
summary.Score = (summary.mean_ARI+summary.mean_MacroF1)/2;
summary = sortrows(summary,{'Score','Delta','Lambda'},{'descend','ascend','ascend'});
writetable(summary,fullfile(outDir,'tuning_grid_summary.csv'));
selected = table(summary.Delta(1),summary.Lambda(1),summary.Score(1), ...
    'VariableNames',{'Delta','Lambda','ValidationScore'});
writetable(selected,fullfile(outDir,'selected_parameters.csv'));

results = run_evaluation(B,Y,c,selected.Delta,selected.Lambda,cfg);
writetable(results,fullfile(outDir,'per_seed_results.csv'));
conditionMeans = groupsummary(results,'Tau','mean',{'ARI','MacroF1','SmallF1'});
writetable(conditionMeans,fullfile(outDir,'condition_means.csv'));
output = struct('Dataset',dataset,'Config',cfg,'GraphInfo',graphInfo, ...
    'SelectedParameters',selected,'TuningResults',tuning, ...
    'PerSeedResults',results,'ConditionMeans',conditionMeans,'OutputDirectory',outDir);
end

function T = run_tuning(B,Y,c,cfg)
n = numel(cfg.tau)*numel(cfg.tuning_seeds)*numel(cfg.delta_grid)*numel(cfg.lambda_grid);
Tau=zeros(n,1);Seed=zeros(n,1);Delta=zeros(n,1);Lambda=zeros(n,1);
ARI=zeros(n,1);MacroF1=zeros(n,1);SmallF1=zeros(n,1);row=0;
for tau=cfg.tau
    for seed=cfg.tuning_seeds
        Bc=generate_primary_corruption_shared(B,Y,tau,seed);
        for delta=cfg.delta_grid
            for lambda=cfg.lambda_grid
                row=row+1; labels=fit_once(Bc,c,seed,delta,lambda,cfg);
                m=evaluate_real_clustering(Y,labels);
                Tau(row)=tau;Seed(row)=seed;Delta(row)=delta;Lambda(row)=lambda;
                ARI(row)=m.ARI;MacroF1(row)=m.MacroF1;SmallF1(row)=m.SmallF1;
            end
        end
    end
end
T=table(Tau,Seed,Delta,Lambda,ARI,MacroF1,SmallF1);
end

function T = run_evaluation(B,Y,c,delta,lambda,cfg)
n=numel(cfg.tau)*numel(cfg.evaluation_seeds);Tau=zeros(n,1);Seed=zeros(n,1);
ARI=zeros(n,1);MacroF1=zeros(n,1);SmallF1=zeros(n,1);row=0;
for tau=cfg.tau
    for seed=cfg.evaluation_seeds
        row=row+1;Bc=generate_primary_corruption_shared(B,Y,tau,seed);
        labels=fit_once(Bc,c,seed,delta,lambda,cfg);m=evaluate_real_clustering(Y,labels);
        Tau(row)=tau;Seed(row)=seed;ARI(row)=m.ARI;
        MacroF1(row)=m.MacroF1;SmallF1(row)=m.SmallF1;
    end
end
T=table(Tau,Seed,ARI,MacroF1,SmallF1);
end

function labels = fit_once(B,c,seed,delta,lambda,cfg)
rng(safe_seed(20260803+10000*seed+77),'twister');
initial=initial_labels_from_bipartite(B,c,1);
rng(safe_seed(20260803+10000*seed+777),'twister');
labels=rmbgc(B,initial,'UseRobust',true,'UseBalance',true, ...
    'Delta',delta,'Lambda',lambda,'MaxIter',cfg.max_iter,'Tolerance',cfg.tolerance);
end

function [Xs,Y,name] = load_public_dataset(dataFile)
S=load(dataFile);if isfield(S,'Xs'),Xs=S.Xs;elseif isfield(S,'X'),Xs=S.X;else,error('run_RMBGC:Features','MAT file must contain Xs or X.');end
if ~iscell(Xs),Xs={Xs};end;Xs=Xs(:);
if isfield(S,'Y'),Y=S.Y;elseif isfield(S,'label'),Y=S.label;else,error('run_RMBGC:Labels','MAT file must contain Y or label.');end
Y=Y(:);assert(all(cellfun(@(x)size(x,1)==numel(Y),Xs)),'run_RMBGC:ViewRows','View rows disagree with labels.');
assert(all(cellfun(@(x)all(isfinite(nonzeros(x))),Xs))&&all(isfinite(Y)),'run_RMBGC:NonFiniteData','Dataset contains NaN or Inf.');
[~,~,Y]=unique(Y,'sorted');[~,name]=fileparts(dataFile);
end

function s=safe_seed(x)
s=mod(round(double(x)),2^32-1);if s<=0,s=s+1;end
end
