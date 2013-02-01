function samp = sampleCRFTables(data, model, samp, isItemTopic)

if isItemTopic
    numUsers = data.numUsers;
    sampkuM = samp.kuM; sampnuM = samp.nuM; samptuM = samp.tuM;
    exampsByUser = data.exampsByUser;
    residCs = samp.residCs;
    muC = samp.muC; mC = samp.mC;
    gammaM = model.gammaM; betaM = model.betaM; c0 = model.c0;
else
    numUsers = data.numItems;
    sampkuM = samp.kjU; sampnuM = samp.njU; samptuM = samp.tjU;
    exampsByUser = data.exampsByItem;
    residCs = samp.residDs;
    muC = samp.muD; mC = samp.mD;
    gammaM = model.gammaU; betaM = model.betaU; c0 = model.d0;
end

% Sample table assignments for users. 
for uu = 1:numUsers
    kuM = sampkuM{uu}; nuM = sampnuM{uu}; tuM = samptuM{uu}; 
    TuM = length(nuM); % Table count
    
    examps = exampsByUser{uu};
    examps_in_new_table = [];
    for ee_i = 1:length(examps)
        ee = examps(ee_i);
        old_t = tuM(ee_i); % Original table
        
        % Assemble multinomial probability vector       
        mult = zeros(1, TuM + 1);
        for tt = 1:TuM
            mult(tt) = double(nuM(tt)) * exp(- (residCs(ee) - muC(kuM(tt))) ^ 2 * model.invsigmaSqd / 2);
        end
        
        % Iterate all existing dishes to get the probability of assigning to a new dish
        mult(TuM + 1) = betaM * exp(- (residCs(ee) - c0) ^ 2 * model.invsigmaSqd / 2);
        divid = betaM;
        for kk = 1:length(mC)
            mult(TuM + 1) = mult(TuM + 1) + double(mC(kk)) * exp(- (residCs(ee) - muC(kk)) ^ 2 * model.invsigmaSqd / 2);
            divid = divid + double(mC(kk));
        end
        mult(TuM + 1) = gammaM * mult(TuM + 1) / divid;        
        mult_norm = mult / sum(mult); % normalize
        
        % Sample new table assignment from the multinomial, update nuM, tuM, kuM
        new_t = find(mnrnd(1,mult_norm));
        
        if length(new_t) > 1
            display(new_t);display(mult_norm);display(mult);display(divid);
        end
        
        tuM(ee_i) = new_t;
        if new_t > TuM && length(nuM) < new_t
            nuM(new_t) = 1;
        else
            nuM(new_t) = nuM(new_t) + 1;
        end
        nuM(old_t) = nuM(old_t) - 1;        
        
        if new_t > TuM
            examps_in_new_table = [examps_in_new_table, ee]; 
        end        
    end
    
    % After sampling tables for all examples, if new table appeared, sample dish for that table
    % Currently we only 1 more table at max per sampling round
    if length(nuM) > TuM        
        k_new = sampleDish(samp, examps_in_new_table, model, isItemTopic);
        kuM(TuM + 1) = k_new;
    end
    sampkuM{uu} = kuM; sampnuM{uu} = nuM; samptuM{uu} = tuM;
end

if isItemTopic
    samp.kuM = sampkuM; samp.nuM = sampnuM; samp.tuM = samptuM;
else
    samp.kjU = sampkuM; samp.njU = sampnuM; samp.tjU = samptuM;
end