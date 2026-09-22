function [B,y,Xinit,ref]=build_mechanism_instance(cfg,seed,p)
    %BUILD_MECHANISM_INSTANCE Recreate one deterministic synthetic experiment.
    % True labels generate and evaluate the experiment but are not used by the clustering optimizer.

    rng(42+100000*seed+round(1e4*p),'twister');
    n3=round(cfg.n*p); n1=floor((cfg.n-n3)/2); n2=cfg.n-n3-n1;
    y=[ones(n1,1);2*ones(n2,1);3*ones(n3,1)];

    mu=zeros(3,cfg.d); mu(2,1)=cfg.separation; mu(3,2)=cfg.separation;
    X=[cfg.latentSD*randn(n1,cfg.d)+mu(1,:); ...
        cfg.latentSD*randn(n2,cfg.d)+mu(2,:); ...
        cfg.latentSD*randn(n3,cfg.d)+mu(3,:)];
    B=cell(cfg.V,1); Xinit=[];
    for v=1:cfg.V
        [Q,~]=qr(randn(cfg.d));
        Z=X*Q+cfg.viewNoiseSD*randn(size(X));
        Xinit=[Xinit Z]; %#ok<AGROW>
        if isfield(cfg,'anchorCounts'), mByCluster=cfg.anchorCounts; else, mByCluster=[34 33 33]; end
        assert(numel(mByCluster)==3 && sum(mByCluster)==cfg.m,'Invalid anchorCounts in cfg.'); a=[];
        for k=1:3
            [~,ak]=kmeans(Z(y==k,:),mByCluster(k),'Start','plus','Replicates',5, ...
                'MaxIter',200,'EmptyAction','singleton','Display','off');
            a=[a;ak]; %#ok<AGROW>
        end
        D=EuDist2(Z,a,0);
        if isfield(cfg,'graphK'), graphK=cfg.graphK; else, graphK=5; end
        graphK=min(graphK,size(D,2)-1);
        assert(graphK>=1 && graphK<size(D,2), ...
            'graphK must leave at least one non-neighbor anchor.');
        h=max(cfg.bandwidthMultiplier*median(sqrt(min(D,[],2))),1e-6);
        [nearestD,nearestAnchor]=mink(D,graphK,2);
        weights=exp(-nearestD/(2*h^2));
        weights=max(weights,realmin('double'));
        weights=weights./sum(weights,2);
        rows=repmat((1:cfg.n)',graphK,1);
        B{v}=sparse(rows,nearestAnchor(:),weights(:),cfg.n,size(D,2),cfg.n*graphK);
        support=full(sum(spones(B{v}),2));
        assert(all(support==graphK) && max(abs(full(sum(B{v},2))-1))<1e-10, ...
            'Top-k bipartite graph construction violated its row invariants.');
    end

    ref=cell(cfg.V,3);
    for v=1:cfg.V
        for k=1:3
            ref{v,k}=mean(B{v}(y==k,:),1);
        end
    end
end
