function samp = sampleCRFDishs(samp, data, model, isItemTopic)

if isItemTopic
    numUsers = data.numUsers;
    sampkuM = samp.kuM; sampnuM = samp.nuM; samptuM = samp.tuM;
    exampsByUser = data.exampsByUser;
else
    numUsers = data.numItems;
    sampkuM = samp.kjU; sampnuM = samp.njU; samptuM = samp.tjU;
    exampsByUser = data.exampsByItem;
end

parfor uu = 1:numUsers
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
        kuM(tt) = sampleDish(samp, tuM_cell{tt}, model, isItemTopic);        
    end
    
    sampkuM{uu} = kuM;
end


if isItemTopic
    samp.kuM = sampkuM;
else
    samp.kjU = sampkuM;
end