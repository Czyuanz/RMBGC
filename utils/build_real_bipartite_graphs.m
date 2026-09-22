function [B, graphInfo] = build_real_bipartite_graphs(Xs,nClusters,opts)
%BUILD_REAL_BIPARTITE_GRAPHS Construct multi-view bipartite graphs from feature matrices.
    arguments
        Xs cell
        nClusters (1,1) double {mustBeInteger,mustBePositive}
        opts.NAnchors (1,1) double {mustBeInteger,mustBeNonnegative} = 0
        opts.NNeighbors (1,1) double {mustBeInteger,mustBePositive} = 5
        opts.AnchorSeed (1,1) double {mustBeInteger} = 20260803
        opts.AnchorMaxIter (1,1) double {mustBeInteger,mustBePositive} = 100
        opts.AnchorReplicates (1,1) double {mustBeInteger,mustBePositive} = 3
        opts.AnchorMethod (1,1) string = "kmeans"
        opts.DenseKMeansMaxBytes (1,1) double {mustBePositive} = 1e9
    end

    n = size(Xs{1},1); V = numel(Xs);
    if opts.NAnchors == 0
        nAnchors = min([5*nClusters,500,n-1]);
    else
        nAnchors = min(opts.NAnchors,n-1);
    end
    nNeighbors = min(opts.NNeighbors,nAnchors-1);
    if nNeighbors < 1, error('build_real_bipartite_graphs:Neighbors','At least two anchors are required.'); end

    rng(opts.AnchorSeed,'twister');
    sharedAnchorRows = randperm(n,nAnchors);
    B = cell(V,1); anchorRows = cell(V,1); featureDims = zeros(V,1);
    for v = 1:V
        X = double(Xs{v});
        if size(X,1) ~= n, error('build_real_bipartite_graphs:Rows','Views must share sample rows.'); end
        if any(~isfinite(nonzeros(X))), error('build_real_bipartite_graphs:Finite','View %d has NaN/Inf.',v); end
        rowNorm = sqrt(full(sum(X.^2,2)));
        X = spdiags(1./max(rowNorm,eps),0,n,n) * X;
        if opts.AnchorMethod=="kmeans" && issparse(X)
            denseBytes=8*numel(X);
            if denseBytes>opts.DenseKMeansMaxBytes
                error('build_real_bipartite_graphs:SparseKMeansMemory', ...
                    ['View %d needs %.2f GB for dense k-means, exceeding the ', ...
                     '%.2f GB safety limit. Use sampled anchors.'], ...
                    v,denseBytes/1e9,opts.DenseKMeansMaxBytes/1e9);
            end
            X=full(X);
        end
        featureDims(v) = size(X,2);
        start = X(sharedAnchorRows,:);
        switch lower(opts.AnchorMethod)
            case "kmeans"
                % The first replicate is the fixed sampled start; additional
                % replicates use k-means++ under the same AnchorSeed stream.
                starts = cell(opts.AnchorReplicates,1); starts{1}=start;
                for rr=2:opts.AnchorReplicates
                    [~,starts{rr}] = kmeans(X,nAnchors,'Start','plus','Replicates',1, ...
                        'MaxIter',opts.AnchorMaxIter,'EmptyAction','singleton','Display','off');
                end
                bestSum=inf; anchors=start;
                for rr=1:opts.AnchorReplicates
                    [~,candidate,sumd] = kmeans(X,nAnchors,'Start',starts{rr},'Replicates',1, ...
                        'MaxIter',opts.AnchorMaxIter,'EmptyAction','singleton','Display','off');
                    if sum(sumd)<bestSum, bestSum=sum(sumd); anchors=candidate; end
                end
                anchorRows{v} = sharedAnchorRows;
            case "sample"
                anchors = start;
                anchorRows{v} = sharedAnchorRows;
            otherwise
                error('build_real_bipartite_graphs:AnchorMethod','Unknown anchor method %s.',opts.AnchorMethod);
        end
        B{v} = ConstructBP_pkn(X,anchors,'nNeighbor',nNeighbors);
        rowSum = full(sum(B{v},2));
        bad = rowSum <= eps;
        if any(bad)
            B{v}(bad,:) = 1/nAnchors;
            rowSum = full(sum(B{v},2));
        end
        B{v} = spdiags(1./rowSum,0,n,n) * B{v};
    end
    graphInfo = struct('nSamples',n,'nViews',V,'nClusters',nClusters, ...
        'nAnchors',nAnchors,'nNeighbors',nNeighbors,'anchorSeed',opts.AnchorSeed, ...
        'anchorMethod',char(opts.AnchorMethod),'anchorRows',{anchorRows},'featureDims',featureDims, ...
        'anchorMaxIter',opts.AnchorMaxIter,'anchorReplicates',opts.AnchorReplicates, ...
        'anchorStart','fixed sample plus k-means++ candidates');
end
