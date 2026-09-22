function [labels, model] = rmbgc(B, initialLabels, opts)
%RMBGC Robust multi-bipartite graph clustering.
%   The method optimizes the following two-term objective:
%
%     min (1/(nV)) sum_iv rho_delta(||b_iv-y_i'C_v||_2)
%         -(lambda/c) log det((c/n)Y'Y).
%
%   Setting UseRobust=false replaces pseudo-Huber by 0.5*u^2. Setting
%   UseBalance=false removes the log-det term. Every feasible solution keeps
%   all clusters nonempty; log-det adds soft repulsion from low occupancy.

    arguments
        B cell
        initialLabels (:,1) double
        opts.UseRobust (1,1) logical = true
        opts.UseBalance (1,1) logical = true
        opts.Lambda (1,1) double {mustBeNonnegative} = 0.01
        opts.Delta (1,1) double {mustBePositive} = 1
        opts.MaxIter (1,1) double {mustBeInteger,mustBePositive} = 50
        opts.Tolerance (1,1) double {mustBePositive} = 1e-6
        opts.RandomOrder (1,1) logical = true
        opts.StoreHistory (1,1) logical = false
    end

    validate_graphs(B);
    labels = initialLabels(:);
    n = size(B{1},1);
    if numel(labels) ~= n
        error('rmbgc:LabelSize','Initial labels must have one entry per sample.');
    end
    [~,~,labels] = unique(labels,'stable');
    initialLabelsCanonical = labels;
    c = max(labels);
    counts = accumarray(labels,1,[c,1]);
    if any(counts == 0) || c < 2
        error('rmbgc:Initialization','Initialization must contain every cluster.');
    end

    C = update_prototypes(B,labels,c,[]);
    initialResidual = assigned_residuals(B,labels,C);
    initialPseudoHuberWeights = 1 ./ sqrt(1 + ...
        (initialResidual./opts.Delta).^2);
    previousObjective = inf;
    trace = nan(opts.MaxIter,6);
    if opts.StoreHistory
        labelHistory = zeros(n,opts.MaxIter);
        prototypeHistory = cell(opts.MaxIter,1);
    end

    for iter = 1:opts.MaxIter
        if opts.UseRobust
            assigned = assigned_residuals(B,labels,C);
            irlsWeights = 1 ./ sqrt(1 + (assigned./opts.Delta).^2);
        else
            irlsWeights = [];
        end
        C = update_prototypes(B,labels,c,irlsWeights);
        dataCost = all_assignment_costs(B,C,opts.UseRobust,opts.Delta);
        [labels,changed] = coordinate_assign(labels,dataCost,opts.UseBalance,opts.Lambda,opts.RandomOrder);

        if opts.UseRobust
            assigned = assigned_residuals(B,labels,C);
            irlsWeights = 1 ./ sqrt(1 + (assigned./opts.Delta).^2);
        else
            irlsWeights = [];
        end
        C = update_prototypes(B,labels,c,irlsWeights);
        [objectiveValue,dataTerm,balanceTerm] = objective_value(B,labels,C,opts);
        counts = accumarray(labels,1,[c,1]);
        trace(iter,:) = [iter,objectiveValue,dataTerm,balanceTerm,changed,min(counts)];
        if opts.StoreHistory
            labelHistory(:,iter) = labels;
            prototypeHistory{iter} = C;
        end

        relativeChange = abs(previousObjective-objectiveValue) / max(1,abs(previousObjective));
        if changed == 0 || (iter > 1 && relativeChange < opts.Tolerance)
            break;
        end
        previousObjective = objectiveValue;
    end

    trace = trace(1:iter,:);
    model = struct();
    model.prototypes = C;
    model.delta = opts.Delta;
    model.lambda = opts.Lambda * double(opts.UseBalance);
    model.useRobust = opts.UseRobust;
    model.useBalance = opts.UseBalance;
    model.objective = trace(end,2);
    model.dataTerm = trace(end,3);
    model.balanceTerm = trace(end,4);
    model.iterations = iter;
    model.counts = accumarray(labels,1,[c,1]);
    model.minClusterFraction = min(model.counts)/n;
    model.predictedSizeCV = std(model.counts/n)/max(mean(model.counts/n),eps);
    model.labelDrift = mean(labels ~= initialLabelsCanonical);
    finalResidual = assigned_residuals(B,labels,C);
    model.initialResidual = initialResidual;
    model.initialPseudoHuberWeights = initialPseudoHuberWeights;
    model.finalResidual = finalResidual;
    if opts.UseRobust
        model.finalIrlsWeights = 1 ./ sqrt(1 + (finalResidual./opts.Delta).^2);
    else
        model.finalIrlsWeights = ones(n,numel(B));
    end
    model.meanIrlsWeight = mean(model.finalIrlsWeights,'all');
    model.balanceContribution = double(opts.UseBalance)*opts.Lambda*model.balanceTerm;
    model.trace = array2table(trace,'VariableNames', ...
        {'Iteration','Objective','DataTerm','BalanceTerm','Changed','MinClusterSize'});
    if opts.StoreHistory
        model.labelHistory = labelHistory(:,1:iter);
        model.prototypeHistory = prototypeHistory(1:iter);
    end
end

function validate_graphs(B)
    if isempty(B), error('rmbgc:EmptyInput','B must contain at least one view.'); end
    n = size(B{1},1);
    for v = 1:numel(B)
        if ~isnumeric(B{v}) || size(B{v},1) ~= n
            error('rmbgc:GraphSize','All bipartite graphs must share the same rows.');
        end
        if any(~isfinite(nonzeros(B{v}))) || any(nonzeros(B{v}) < 0)
            error('rmbgc:GraphValues','Bipartite graphs must be finite and nonnegative.');
        end
        rowSum = full(sum(B{v},2));
        if max(abs(rowSum-1)) > 1e-7
            error('rmbgc:RowStochastic','Every bipartite-graph row must sum to one.');
        end
    end
end

function C = update_prototypes(B,labels,c,weights)
    n = numel(labels); V = numel(B); C = cell(V,1);
    for v = 1:V
        if isempty(weights), w = ones(n,1); else, w = weights(:,v); end
        membership = sparse((1:n)',labels,w,n,c,n);
        mass = full(sum(membership,1))';
        if any(mass <= 0), error('rmbgc:ZeroMass','A prototype received zero IRLS mass.'); end
        C{v} = spdiags(1./mass,0,c,c) * (membership' * B{v});
        C{v} = bsxfun(@rdivide,C{v},max(sum(C{v},2),eps));
    end
end

function r = assigned_residuals(B,labels,C)
    n = numel(labels); V = numel(B); r = zeros(n,V);
    for v = 1:V
        Bv = B{v}; Cv = C{v};
        aa = full(sum(Bv.^2,2));
        cc = full(sum(Cv.^2,2));
        cross = full(sum(Bv .* Cv(labels,:),2));
        r(:,v) = sqrt(max(aa + cc(labels) - 2*cross,0));
    end
end

function cost = all_assignment_costs(B,C,useRobust,delta)
    n = size(B{1},1); c = size(C{1},1); V = numel(B);
    cost = zeros(n,c);
    for v = 1:V
        Bv = B{v}; Cv = C{v};
        aa = full(sum(Bv.^2,2)); bb = full(sum(Cv.^2,2))';
        distance2 = max(bsxfun(@plus,aa,bb) - 2*full(Bv*Cv'),0);
        u = sqrt(distance2);
        if useRobust
            cost = cost + delta^2 * (sqrt(1 + (u./delta).^2) - 1);
        else
            cost = cost + 0.5*u.^2;
        end
    end
    cost = cost / V;
end

function [labels,changed] = coordinate_assign(labels,dataCost,useBalance,lambda,randomOrder)
    n = numel(labels); c = size(dataCost,2);
    counts = accumarray(labels,1,[c,1]); changed = 0;
    if randomOrder, order = randperm(n); else, order = 1:n; end
    for q = 1:n
        i = order(q); old = labels(i); score = dataCost(i,:);
        if counts(old) <= 1
            score(:) = inf; score(old) = dataCost(i,old);
        elseif useBalance && lambda > 0
            candidates = 1:c; candidates(old) = [];
            ratio = ((counts(old)-1) .* (counts(candidates)+1)) ./ ...
                (counts(old) .* counts(candidates));
            score(candidates) = score(candidates) - (n*lambda/c)*log(ratio)';
        end
        [~,best] = min(score);
        if best ~= old
            counts(old) = counts(old)-1; counts(best) = counts(best)+1;
            labels(i) = best; changed = changed+1;
        end
    end
end

function [value,dataTerm,balanceTerm] = objective_value(B,labels,C,opts)
    allCost = all_assignment_costs(B,C,opts.UseRobust,opts.Delta);
    idx = sub2ind(size(allCost),(1:numel(labels))',labels);
    dataTerm = mean(allCost(idx));
    counts = accumarray(labels,1,[size(C{1},1),1]);
    balanceTerm = -mean(log((numel(counts)/numel(labels))*counts));
    value = dataTerm + double(opts.UseBalance)*opts.Lambda*balanceTerm;
end
