function report = run_mechanism_experiment(mode,numWorkers,outputDir)
%RUN_MECHANISM_EXPERIMENT Compute the Base collapse-probability grid.
if nargin<1,mode="smoke";end
if nargin<2||isempty(numWorkers),numWorkers=4;end
root=fileparts(mfilename('fullpath'));
if nargin<3||strlength(string(outputDir))==0,outputDir=fullfile(root,'results','mechanism_base');end
addpath(root,fullfile(root,'utils'),fullfile(root,'measures'));
mode=string(mode);assert(ismember(mode,["smoke","formal"]),'run_mechanism_experiment:Mode');
validateattributes(numWorkers,{'numeric'},{'scalar','integer','positive'});
cfg=mechanism_config();if mode=="smoke",seeds=11;else,seeds=11:210;end
if ~isfolder(outputDir),mkdir(outputDir);end
cacheDir=fullfile(outputDir,'checkpoints');if ~isfolder(cacheDir),mkdir(cacheDir);end
pool=gcp('nocreate');
if ~isempty(pool)&&pool.NumWorkers~=numWorkers,delete(pool);pool=[];end
if isempty(pool),parpool('Processes',numWorkers);end
parts=cell(numel(seeds),1);
parfor si=1:numel(seeds)
    seed=seeds(si);file=fullfile(cacheDir,sprintf('seed_%03d.mat',seed));
    if isfile(file),S=load(file,'T');parts{si}=S.T;else,T=run_seed(seed,cfg);save_checkpoint(file,T);parts{si}=T;end
end
T=sortrows(vertcat(parts{:}),{'Pi','Epsilon','Seed'});
writetable(T,fullfile(outputDir,'base_mechanism_per_seed.csv'));
summary=groupsummary(T,{'Pi','Epsilon','Chi'},'mean',{'Collapse','TargetF1','CleanTargetF1'});
writetable(summary,fullfile(outputDir,'base_collapse_probability.csv'));
report=struct('Mode',mode,'Seeds',seeds,'Rows',height(T),'ExpectedRows',67*numel(seeds), ...
    'Complete',height(T)==67*numel(seeds),'OutputDirectory',string(outputDir));
assert(report.Complete,'run_mechanism_experiment:Incomplete');
end

function T=run_seed(seed,cfg)
T=table();target=3;
for pi=cfg.pi_grid
    [Bclean,Y]=build_mechanism_instance(cfg.synthetic,seed,pi);
    n=numel(Y);nTarget=sum(Y==target);initSeed=safe_seed(310000+seed+round(1e5*pi));
    rng(initSeed,'twister');cleanInit=initial_labels_from_bipartite(Bclean,3,1);
    clean=fit_base(Bclean,cleanInit);cleanTarget=best_target_cluster(Y,clean.labels,target,3);
    [~,~,cleanF1]=prf(Y==target,clean.labels==cleanTarget);
    for epsilon=cfg.epsilon_grid
        if epsilon>=pi,continue,end
        b=round(n*epsilon);
        if b==0
            Bcorr=Bclean;
        else
            attackSeed=safe_seed(410000+seed*1009+round(1e5*pi)+round(1e6*epsilon)*97);
            Bcorr=apply_class_target_anchor_mass_hijacking(Bclean,Y,target,b,attackSeed, ...
                'Gamma',.90,'ReplaceRatio',.125,'ViewFraction',.50);
        end
        rng(initSeed,'twister');corrInit=initial_labels_from_bipartite(Bcorr,3,1);
        corr=fit_base(Bcorr,corrInit);map=align_clusters(clean.labels,corr.labels,3);corrTarget=map(cleanTarget);
        [~,~,targetF1]=prf(Y==target,corr.labels==corrTarget);collapse=double(targetF1<.7*cleanF1);
        T=[T;table(pi,epsilon,epsilon/pi,seed,b,nTarget,cleanF1,targetF1,collapse, ...
            'VariableNames',{'Pi','Epsilon','Chi','Seed','CorruptedSamples','TargetSize', ...
            'CleanTargetF1','TargetF1','Collapse'})]; %#ok<AGROW>
    end
end
end

function out=fit_base(B,initial)
rng(safe_seed(610000+size(B{1},1)),'twister');
[labels,model]=rmbgc(B,initial,'UseRobust',false,'UseBalance',false, ...
    'Delta',1,'Lambda',0,'MaxIter',50,'Tolerance',1e-6);
out=struct('labels',labels,'model',model);
end

function k=best_target_cluster(Y,labels,target,c)
scores=zeros(c,1);for j=1:c,[~,~,scores(j)]=prf(Y==target,labels==j);end
[~,k]=max(scores);
end

function map=align_clusters(cleanLabels,corrLabels,c)
overlap=zeros(c);for i=1:c,for j=1:c,overlap(i,j)=sum(cleanLabels==i&corrLabels==j);end,end
[map,~]=hungarian(max(overlap(:))-overlap);
end

function [precision,recall,f1]=prf(truth,prediction)
tp=sum(truth&prediction);precision=tp/max(sum(prediction),1);recall=tp/max(sum(truth),1);
f1=2*precision*recall/max(precision+recall,eps);
end

function cfg=mechanism_config()
cfg.pi_grid=[.02 .03 .04 .05 .075 .10 .15 .20];
cfg.epsilon_grid=[0 .0025 .005 .01 .015 .02 .03 .04 .06 .08 .12];
cfg.synthetic=struct('n',600,'d',8,'V',3,'m',30,'separation',2.5,'latentSD',1, ...
    'viewNoiseSD',.35,'bandwidthMultiplier',1,'anchorCounts',[10 10 10],'graphK',8);
end

function s=safe_seed(x)
s=mod(round(double(x))-1,2^31-2)+1;
end

function save_checkpoint(file,T)
save(file,'T','-v7.3');
end
