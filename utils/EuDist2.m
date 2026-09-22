function D = EuDist2(fea_a, fea_b, bSqrt)
    %EUDIST2 Pairwise Euclidean distances, reused from CARD-main/utils.
    if nargin < 3, bSqrt = 1; end
    if nargin < 2 || isempty(fea_b), fea_b = fea_a; end
    aa = sum(fea_a .* fea_a, 2);
    bb = sum(fea_b .* fea_b, 2);
    D = bsxfun(@plus, aa, bb') - 2 * (fea_a * fea_b');
    D = max(D, 0);
    if bSqrt, D = sqrt(D); end
end
