function k_new = sampleDish(samp, examps, model, isItemTopic)

% When sampling for an empty table, return 1 simply.
if isempty(examps) 
    k_new = 1;
    return;
end

if isItemTopic
    mC = samp.mC; muC = samp.muC; kU = samp.kU; muD = samp.muD; nC = samp.nC; nD = samp.nD;
else
    mC = samp.mD; muC = samp.muD; kU = samp.kM; muD = samp.muC; nC = samp.nD; nD = samp.nC;
end

k_new = sampleDishFull(examps,samp.resids,mC,muC,muD,kU,nC, nD, model,isItemTopic);
