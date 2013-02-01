function samp = sampleCRFTablesRealTime(data, model, samp, isItemTopic)

tStart = tic;
if isItemTopic
    numUsers = data.numUsers;
    sampkuM = samp.kuM; sampnuM = samp.nuM; samptuM = samp.tuM;
    exampsByUser = data.exampsByUser;
    muC = samp.muC; muD = samp.muD; mC = samp.mC; nC = samp.nC; tM = samp.tM; kM = samp.kM; kU = samp.kU;
    gammaM = model.gammaM; betaM = model.betaM; c0 = model.c0;
else
    numUsers = data.numItems;
    sampkuM = samp.kjU; sampnuM = samp.njU; samptuM = samp.tjU;
    exampsByUser = data.exampsByItem;
    muC = samp.muD; muD = samp.muC; mC = samp.mD; nC = samp.nD; tM = samp.tU; kM = samp.kU; kU = samp.kM;
    gammaM = model.gammaU; betaM = model.betaU; c0 = model.d0;
end

% Sample table assignments for users. 
for uu = 1:numUsers
    kuM = sampkuM{uu}; nuM = sampnuM{uu}; tuM = samptuM{uu}; 
    TuM = length(nuM); % Table count
    
    examps = exampsByUser{uu};
    examps_in_new_table = []; 
    create_new_table = false;
    
    % Find an old empty table in advance
    empty_tables = find(nuM==0);
    new_table_index = TuM + 1;
    if ~isempty(empty_tables)
        new_table_index = empty_tables(1);
    end
    
    for ee_i = 1:length(examps)
        ee = examps(ee_i);
        old_t = tuM(ee_i); % Original table
        residC = samp.resids(ee) - muD(kU(ee));
        
        if isItemTopic == false && ee == 1809
            ee;
        end
        
        % Assemble multinomial probability vector       
        mult = zeros(1, TuM + 1);
        for tt = 1:TuM
            mult(tt) = double(nuM(tt)) * exp(- (residC - muC(kuM(tt))) ^ 2 * model.invsigmaSqd / 2);
        end
        
        % Iterate all existing dishes to get the probability of assigning to a new dish
        mult(TuM + 1) = betaM * exp(- (residC - c0) ^ 2 * model.invsigmaSqd / 2);
        divid = betaM;
        for kk = 1:length(mC)
            mult(TuM + 1) = mult(TuM + 1) + double(mC(kk)) * exp(- (residC - muC(kk)) ^ 2 * model.invsigmaSqd / 2);
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
        
        if new_t > TuM
            create_new_table = true;
            new_t = new_table_index;
        end
                
        tuM(ee_i) = new_t;
        if create_new_table && length(nuM) < new_t
            nuM(new_t) = 0;
        end
        
        nuM(new_t) = nuM(new_t) + 1;
        nuM(old_t) = nuM(old_t) - 1;   
        
        % Remove table as needed
        if nuM(old_t) == 0
            mC(kuM(old_t)) = mC(kuM(old_t)) - 1;
        end
        
        if create_new_table
            examps_in_new_table = [examps_in_new_table, ee]; 
        end
    end
    
    % After sampling tables for all examples, if new table appeared, sample dish for that table
    % Currently we only 1 more table at max per sampling round
    if create_new_table
        k_new = sampleDish(samp, examps_in_new_table, model, isItemTopic);
        kuM(new_table_index) = k_new;
        
        % Add table as needed
        if k_new > length(mC) % New dish
            mC(k_new) = 0;
        end
        mC(k_new) = mC(k_new) + 1;
    end
    
    % Update dish sufficient statisticcs at once
    for ee_i = 1:length(examps)
        ee = examps(ee_i); 
        old_t = tM(ee); old_k = kuM(old_t);
        new_t = tuM(ee_i); new_k = kuM(new_t); 
        if new_k > length(nC)
            muC(new_k) = 0; nC(new_k) = 0; 
        end
        tM(ee) = new_t; kM(ee) = new_k;
        residC = samp.resids(ee) - muD(kU(ee));
        muC(new_k) = (muC(new_k) * nC(new_k) + residC) / (nC(new_k) + 1); nC(new_k) = nC(new_k) + 1; 
        if nC(old_k) == 1
            muC(old_k) = 0; nC(old_k) = 0;
        else
            muC(old_k) = (muC(old_k) * nC(old_k) - residC) / (nC(old_k) - 1); nC(old_k) = nC(old_k) - 1;
        end
    end
    
    sampkuM{uu} = kuM; sampnuM{uu} = nuM; samptuM{uu} = tuM;
end

if isItemTopic
    samp.kuM = sampkuM; samp.nuM = sampnuM; samp.tuM = samptuM; samp.tM = tM; samp.kM = kM; samp.muC = muC; samp.nC = nC; samp.mC = mC;
else
    samp.kjU = sampkuM; samp.njU = sampnuM; samp.tjU = samptuM; samp.tU = tM; samp.kU = kM; samp.muD = muC; samp.nD = nC; samp.mD = mC;
end


fprintf('\tsampleCRFTablesRealTime------%g\n',toc(tStart));