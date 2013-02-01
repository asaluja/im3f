function samp = sampleCRFDishsRealTime(samp, data, model, isItemTopic, iscollapsed)

tStart = tic;

if isItemTopic
    numUsers = data.numUsers;
    sampkuM = samp.kuM; sampnuM = samp.nuM; samptuM = samp.tuM;
    exampsByUser = data.exampsByUser;
    muC = samp.muC; muD = samp.muD; mC = samp.mC; nC = samp.nC; nD = samp.nD; kM = samp.kM; kU = samp.kU;
    c = samp.c; d = samp.d;
else
    numUsers = data.numItems;
    sampkuM = samp.kjU; sampnuM = samp.njU; samptuM = samp.tjU;
    exampsByUser = data.exampsByItem;
    muC = samp.muD; muD = samp.muC; mC = samp.mD; nC = samp.nD; nD = samp.nC; kM = samp.kU; kU = samp.kM;
    c = samp.d; d = samp.c;
end

for uu = 1:numUsers
    kuM = sampkuM{uu}; nuM = sampnuM{uu}; tuM = samptuM{uu};    
    TuM = length(nuM);
    
    % First assemble all examps for each table
    examps = exampsByUser{uu};
    tuM_cell = cell(1, TuM);
    for ee_i = 1:length(tuM)
        tuM_cell{tuM(ee_i)} = [tuM_cell{tuM(ee_i)}, examps(ee_i)];
    end
    
    % Sample a new dish for each table
    for tt = 1:TuM
        if isempty(tuM_cell{tt})
            continue;
        end
        
        old_k = kuM(tt);
        % Update global dish sufficient statisticcs at once
        mC(old_k) = mC(old_k) - 1;
        for ee = tuM_cell{tt}
            if strcmp(iscollapsed, 'collapsed')
                residC = samp.resids(ee) - muD(kU(ee));
            else % Non collapsed
                residC = samp.resids(ee) - d(kU(ee));
            end
            muC(old_k) = updateCRFMu(muC, nC, residC, old_k, false, model); nC(old_k) = nC(old_k) - 1;
        end
        
        %new_k = sampleDish(samp, tuM_cell{tt}, model, isItemTopic);
        new_k = sampleDishFull(tuM_cell{tt},samp.resids,mC,muC,muD,kU,nC, nD, model,isItemTopic);
        if length(new_k) > 1
            display(uu);display(kuM);display(nuM);display(tuM);display(TuM);
            display(tt);display(old_k);display(new_k);display(tuM_cell{tt});
            new_k = randi(length(mC) + 1);
        end
        empty_dishes = find(nC == 0);
        if ~isempty(empty_dishes)
            new_k = empty_dishes(1);
        end
        kuM(tt) = new_k;
        
        if new_k > length(mC)
            mC(new_k) = 0;
        end
        mC(new_k) = mC(new_k) + 1;
        
        % Update dish sufficient statisticcs at once
        for ee = tuM_cell{tt}
            if new_k > length(nC)
                muC(new_k) = model.d0 * model.sigmaSqd / model.sigmaSqd0; nC(new_k) = 0; 
            end
            kM(ee) = new_k;
            if strcmp(iscollapsed, 'collapsed')
                residC = samp.resids(ee) - muD(kU(ee));
            else % Non collapsed
                residC = samp.resids(ee) - d(kU(ee));
            end
            muC(new_k) = updateCRFMu(muC, nC, residC, new_k, true, model); nC(new_k) = nC(new_k) + 1;
        end
    end
    
    
    sampkuM{uu} = kuM;
end


if isItemTopic
    samp.kuM = sampkuM; samp.kM = kM; samp.muC = muC; samp.nC = nC; samp.mC = mC;
else
    samp.kjU = sampkuM; samp.kU = kM; samp.muD = muC; samp.nD = nC; samp.mD = mC;
end


fprintf('\tsampleCRFDishesRealTime------%g\n',toc(tStart));