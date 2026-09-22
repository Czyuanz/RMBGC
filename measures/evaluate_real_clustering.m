function metrics = evaluate_real_clustering(truth,prediction)
%EVALUATE_REAL_CLUSTERING ARI, Macro-F1, and bottom-30% Small-F1.
    truth = truth(:); prediction = prediction(:);
    if numel(truth) ~= numel(prediction), error('evaluate_real_clustering:Size','Label sizes differ.'); end
    mapped = best_map(truth,prediction);
    classes = unique(truth); c = numel(classes); f1 = zeros(c,1); precision = zeros(c,1);
    recall = zeros(c,1); counts = zeros(c,1);
    for k = 1:c
        label = classes(k); actual = truth == label; predicted = mapped == label;
        tp = sum(actual & predicted); fp = sum(~actual & predicted); fn = sum(actual & ~predicted);
        precision(k) = tp/max(tp+fp,1); recall(k) = tp/max(tp+fn,1);
        f1(k) = 2*precision(k)*recall(k)/max(precision(k)+recall(k),eps); counts(k) = sum(actual);
    end
    [~,order] = sortrows([counts,double(classes)],[1,2]);
    small = order(1:ceil(0.3*c));
    metrics = struct('ARI',computeARI(truth,prediction),'MacroF1',mean(f1), ...
        'SmallPrecision',mean(precision(small)),'SmallRecall',mean(recall(small)), ...
        'SmallF1',mean(f1(small)),'PerClassPrecision',precision, ...
        'PerClassRecall',recall,'PerClassF1',f1,'ClassCounts',counts, ...
        'SmallClassLabels',classes(small));
end
