function [Bout,meta] = apply_class_target_anchor_mass_hijacking(B,truth,target,b,seed,opts)
%APPLY_CLASS_TARGET_ANCHOR_MASS_HIJACKING Apply controlled target-class anchor-mass corruption.
% Labels select only the target class and equal absolute budget.  They never
% determine corrupted views or replacement-anchor directions.
arguments
    B cell
    truth (:,1) double
    target (1,1) double
    b (1,1) double {mustBeInteger,mustBePositive}
    seed (1,1) double
    opts.Gamma (1,1) double = .90
    opts.ReplaceRatio (1,1) double = .125
    opts.ViewFraction (1,1) double = .50
end
n=size(B{1},1); V=numel(B); idx=find(truth==target); nTarget=numel(idx);
assert(b<nTarget,'MassCorruption:InvalidBudget','Only b < n_target is legal.');
Bout=B; selected=false(n,V); rng(safe_seed(seed),'twister');
chosen=idx(randperm(nTarget,b)); nViews=max(1,min(V,round(opts.ViewFraction*V)));
for jj=1:b
    i=chosen(jj); views=randperm(V,nViews); selected(i,views)=true;
    for v=views
        old=Bout{v}(i,:); [~,nz,w]=find(old); degree=numel(nz);
        nAnchors=size(old,2);
        h=min([degree-1,nAnchors-degree,max(1,round(opts.ReplaceRatio*degree))]);
        assert(h>=1,'MassCorruption:NoAdmissibleRewire','Row support does not admit a safe rewire.');
        take=randperm(degree,h); available=setdiff(1:nAnchors,nz);
        dst=available(randperm(numel(available),h)); keep=true(degree,1); keep(take)=false;
        keepNz=nz(keep); keepW=w(keep); keepW=(1-opts.Gamma)*keepW/sum(keepW);
        wrongW=opts.Gamma*ones(h,1)/h;
        Bout{v}(i,:)=sparse(1,[keepNz(:);dst(:)],[keepW(:);wrongW(:)],1,size(old,2));
    end
end
for v=1:V
    assert(all(nonzeros(Bout{v})>=0),'MassCorruption:NegativeWeight');
    assert(max(abs(full(sum(Bout{v},2))-1))<1e-10,'MassCorruption:RowSum');
    assert(isequal(full(sum(spones(Bout{v}),2)),full(sum(spones(B{v}),2))), ...
        'MassCorruption:SparsityChanged');
end
meta=struct('target',target,'targetSize',nTarget,'budget',b,'sampleIds',chosen(:), ...
    'selectedMask',selected,'corruptedViews',nViews,'viewFraction',nViews/V, ...
    'gamma',opts.Gamma,'replaceRatio',opts.ReplaceRatio,'localExposure',b/nTarget, ...
    'globalExposure',b/n);
end
function s=safe_seed(x),s=mod(round(x)-1,2^31-2)+1;end
