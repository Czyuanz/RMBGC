function labels = initial_labels_from_bipartite(B,nClusters,replicates)
%INITIAL_LABELS_FROM_BIPARTITE Initialize labels from multi-view bipartite graphs.
    if nargin < 3, replicates = 1; end
    % This is deliberately a direct initialization on the supplied graph
    % rows, not an additional spectral-clustering stage.  View blocks are
    % equally weighted and each sample row is normalized before k-means.
    V = numel(B); blocks = cell(V,1);
    for v = 1:V
        blocks{v}=B{v}/sqrt(V);
    end
    embedding = horzcat(blocks{:});
    rowNorm=sqrt(full(sum(embedding.^2,2)));
    embedding=spdiags(1./max(rowNorm,eps),0,size(embedding,1),size(embedding,1))*embedding;
    labels = kmeans(embedding,nClusters,'Start','plus','Replicates',replicates, ...
        'MaxIter',500,'EmptyAction','singleton','Display','off');
end
