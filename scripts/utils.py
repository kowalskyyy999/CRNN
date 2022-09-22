def AccuracyMetric(gt, pred):
    max_len = max(len(gt), len(pred))
    value = 0
    for x, y in zip(gt, pred):
        if x == y:
            value += 1
    return value / max_len