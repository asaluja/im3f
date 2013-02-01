function samp = sampleCRFTablesRealRealTime(data, model, samp, isItemTopic, iscollapsed)

tStart = tic;
if isItemTopic
    numUsers = data.numUsers;
    sampkuM = samp.kuM; sampnuM = samp.nuM; samptuM = samp.tuM;
    exampsByUser = data.exampsByUser;
    muC = samp.muC; muD = samp.muD; mC = samp.mC; nC = samp.nC; nD = samp.nD; tM = samp.tM; kM = samp.kM; kU = samp.kU; 
    c = samp.c; d = samp.d;
    gammaM = model.gammaM; betaM = model.betaM; c0 = model.c0;
else
    numUsers = data.numItems;
    sampkuM = samp.kjU; sampnuM = samp.njU; samptuM = samp.tjU;
    exampsByUser = data.exampsByItem;
    muC = samp.muD; muD = samp.muC; mC = samp.mD; nC = samp.nD; nD = samp.nC; tM = samp.tU; kM = samp.kU; kU = samp.kM;
    c = samp.d; d = samp.c;
    gammaM = model.gammaU; betaM = model.betaU; c0 = model.d0;
end

% Sample table assignments for users. 
for uu = 1:numUsers
    kuM = sampkuM{uu}; nuM = sampnuM{uu}; tuM = samptuM{uu};     
    examps = exampsByUser{uu};
    
    last_example_got_new_table = true;
        
    for ee_i = 1:length(examps)
        if last_example_got_new_table
            TuM = length(nuM); % Table count
            % Find an old empty table in advance
            empty_tables = find(nuM==0);
            new_table_index = TuM + 1;
            if ~isempty(empty_tables)
                new_table_index = empty_tables(1);
            end
        end
        
        ee = examps(ee_i);
        old_t = tuM(ee_i); % Original table
        if strcmp(iscollapsed, 'collapsed')
            residC = samp.resids(ee) - muD(kU(ee));
        else % Non collapsed
            residC = samp.resids(ee) - d(kU(ee));
        end
        % Remove current rating from table
        nuM(old_t) = nuM(old_t) - 1;
        
        % Remove table as needed
        if nuM(old_t) == 0
            mC(kuM(old_t)) = mC(kuM(old_t)) - 1;
        end
        
        % Remove current rating from global sufficient statistics
        old_k = kuM(old_t);            
        muC(old_k) = updateCRFMu(muC, nC, residC, old_k, false, model); nC(old_k) = nC(old_k) - 1;
        
        % Assemble multinomial probability vector       
        mult = zeros(1, TuM + 1);
        for tt = 1:TuM
            sigmaStarSqd = getSigmaStarSqd(kuM(tt), kU(ee), nC, nD, model);
            mult(tt) = double(nuM(tt)) * exp(- (residC - muC(kuM(tt))) ^ 2 / sigmaStarSqd / 2) / (sigmaStarSqd ^ 0.5);
        end
        
        % Iterate all existing dishes to get the probability of assigning to a new dish
        sigmaDSqd = getSigmaDSqd(kU(ee), nD, model);
        % TODO change this sigmaC (= sigma0)
        mult(TuM + 1) = betaM * exp(- (residC - c0) ^ 2 / (model.sigmaSqd + sigmaDSqd) / 2) / ((sigmaDSqd + model.sigmaSqd) ^ 0.5);
        divid = betaM;
        for kk = 1:length(mC)            
            sigmaStarSqd = getSigmaStarSqd(kk, kU(ee), nC, nD, model);
            mult(TuM + 1) = mult(TuM + 1) + double(mC(kk)) * exp(- (residC - muC(kk)) ^ 2 / sigmaStarSqd / 2) / (sigmaStarSqd ^ 0.5);
            divid = divid + double(mC(kk));
        end
        mult(TuM + 1) = gammaM * mult(TuM + 1) / divid;        
        mult_norm = mult / sum(mult); % normalize
        
        % Sample new table assignment from the multinomial, update nuM, tuM, kuM
        new_t = find(mnrnd(1,mult_norm));
        
        if length(new_t) > 1
            display(ee_i);display(ee);display(data.users(ee));display(data.items(ee));display(isItemTopic);            
            display(nuM);display(kuM);display(muC(kuM));
            display(new_t);display(mult_norm);display(mult);display(divid);
            display(residC);display(muD);display(kU(ee));
            display(mC);display(muC);           
        end
                
        if length(new_t) > 1
            new_t = randi(TuM + 1); % Hack
        end
        
        tuM(ee_i) = new_t;
        
        if new_t > TuM            
            new_t = new_table_index;
            tuM(ee_i) = new_t;
            nuM(new_t) = 0;
            
            %new_k = sampleDish(samp, ee, model, isItemTopic);
            new_k = sampleDishFull(examps,samp.resids,mC,muC,muD,kU,nC, nD, model,isItemTopic);
            kuM(new_t) = new_k;
            
            % Add table as needed
            if new_k > length(mC) % New dish
                mC(new_k) = 0;
            end
            mC(new_k) = mC(new_k) + 1;
            last_example_got_new_table = true;
        else
            last_example_got_new_table = false;
        end
        
        nuM(new_t) = nuM(new_t) + 1;
        
        tM(ee) = new_t;
        
        % Update dish sufficient statistics
        new_k = kuM(new_t);
        %new_k = kuM(new_t);
        if new_k > length(nC)
            muC(new_k) = model.d0 * model.sigmaSqd / model.sigmaSqd0; nC(new_k) = 0;
        end
        kM(ee) = new_k;
        muC(new_k) = updateCRFMu(muC, nC, residC, new_k, true, model); nC(new_k) = nC(new_k) + 1;    
    end     
    
    sampkuM{uu} = kuM; sampnuM{uu} = nuM; samptuM{uu} = tuM;
end

if isItemTopic
    samp.kuM = sampkuM; samp.nuM = sampnuM; samp.tuM = samptuM; samp.tM = tM; samp.kM = kM; samp.muC = muC; samp.nC = nC; samp.mC = mC;
else
    samp.kjU = sampkuM; samp.njU = sampnuM; samp.tjU = samptuM; samp.tU = tM; samp.kU = kM; samp.muD = muC; samp.nD = nC; samp.mD = mC;
end


fprintf('\tsampleCRFTablesRealRealTime------%g\n',toc(tStart));