function updateMuM(samp, data, zM, u, j, isplus)
% Update muM in the sample according to the assignment of the given
% rating's item topic

if isplus % Add to muM
    samp.sum_muM(zM) += 