function Z = ConstructBP_pkn(X, anchors, varargin)
    %CONSTRUCTBP_PKN Sparse adaptive-neighbor bipartite graph (reused from CARD).
    % Adapted from CARD-main/utils/ConstructBP_pkn.m, originally associated
    % with Clustering and Projected Clustering with Adaptive Neighbors (KDD 2014).
    [nSmp, ~] = size(X);
    nAnchor = size(anchors,1);
    [eid, errmsg, nNeighbor] = getargs({'nNeighbor'}, {5}, varargin{:});
    if ~isempty(eid), error('ConstructBP_pkn:%s', errmsg); end
    nNeighbor = min(nNeighbor, nAnchor-1);
    if nNeighbor < 1, Z = ones(nSmp,1); return; end
    D = EuDist2(X, anchors, 0);
    [D2, Idx] = sort(D, 2, 'ascend');
    v1 = D2(:, nNeighbor+1);
    v2 = D2(:, 1:nNeighbor);
    weights = bsxfun(@times, bsxfun(@minus,v1,v2), ...
        1 ./ max(nNeighbor*v1-sum(v2,2), eps));
    % MATLAB column-major order of idx_k(:) is neighbor-column first, so the
    % sample indices must repeat as complete 1:n blocks (as in CARD).
    row_idx = repmat((1:nSmp)', nNeighbor, 1);
    idx_k = Idx(:,1:nNeighbor);
    Z = sparse(row_idx, idx_k(:), weights(:), nSmp, nAnchor, nSmp*nNeighbor);
end
