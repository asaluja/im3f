function [samp] = m3f_tib_initSamp(topicModel, iscollapsed, model, data)
%M3F_TIB_INITSAMP Return an initial "sample" from model initialized
%according to initMode.
%
% Usage:
%    [samp] = m3f_tib_initSamp(initMode, model, data, testData)
%
% Inputs:
%    topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}
%    initMode - Mode used to create initial sample
%               1 => Draw random sample from model prior
%               2 => Initialize hidden variables to model means
%               3 => Initialize static factors to MAP estimate trained
%               using stochastic gradient descent and set remaining
%               variables to model means
%    model - m3f_tib structure (see m3f_tib_initModel)
%    data - Dyadic data structure (see loadDyadicData)
%    testData - Dyadic test data structure (see loadDyadicData)
%
% Outputs:
%    samp - Populated sample structure with fields corresponding to
%           the hidden parameters and topics of the m3f_tib:
%           LambdaU (numFacs x numFacs)
%           LambdaM (numFacs x numFacs)
%           muU (numFacs x 1)
%           muM (numFacs x 1)
%           logthetaU (KU x numUsers)
%           a (numFacs x numUsers)
%           c (KM x numUsers)
%           logthetaM (KM x numItems)
%           b (numFacs x numItems)
%           d (KU x numItems)
%           chi (1 x 1)
%           zU (numExamples x 1)
%           zM (numExamples x 1)
%           muC (KM x numUsers or 1)
%           muD (KU x numItems or 1)
%           nC (KM x numUsers or 1)
%           nD (KU x numItems or 1)
%           

% -----------------------------------------------------------------------     



fprintf('Initializing first sample with prior means\n');

% Initialize parameters to model means
samp.LambdaU = model.nu0*model.W0;
samp.LambdaM = model.nu0*model.W0;
samp.muU = model.mu0;
samp.muM = model.mu0;
samp.a = repmat(samp.muU, 1, model.numUsers);
samp.b = repmat(samp.muM, 1, model.numItems);
samp.c = repmat(model.c0, model.KM, model.numUsers);
samp.d = repmat(model.d0, model.KU, model.numItems);
samp.chi = model.chi0;


% Toney
% Initialize muC, muD, nC, nD, KM, KU
% Here, the sample KM, KU are initially same as model KM, KU, but will
% differ as we sample new topics.

KM = model.KM; KU = model.KU;

samp.KM = KM; samp.KU = KU; 

switch topicModel
    case 'secrp'
        samp.muC = repmat(model.c0, KM, data.numUsers);
        samp.muD = repmat(model.c0, KU, data.numItems);
        samp.nC = repmat(0, KM, data.numUsers);
        samp.nD = repmat(0, KU, data.numItems);
        % Sample topics for all examples
        samp.zU = uint32(randi(KU, data.numExamples, 1));
        samp.zM = uint32(randi(KM, data.numExamples, 1));        
        samp = batchUpdate(samp, data, model, topicModel);
    case 'crf'
        samp.muC = repmat(model.c0, 1, KM);     % 1 x KM
        samp.mC = repmat(0, 1, KM);             % 1 x KM
        samp.nC = repmat(0, 1, KM);             % 1 x KM
        samp.kuM = cell(data.numUsers, 1);      % U x 1, 1 x TuM
        samp.nuM = cell(data.numUsers, 1);      % U x 1, 1 x TuM
        samp.tuM = cell(data.numUsers, 1); % Dish assignments for each user's all examples. U x 1, 1 x numExamples(u)
        samp.tM = uint32(repmat(0, 1, data.numExamples));
        samp.kM = uint32(repmat(0, 1, data.numExamples));
        samp.muD = repmat(model.d0, 1, KU);
        samp.mD = repmat(0, 1, KU);
        samp.nD = repmat(0, 1, KU);
        samp.kjU = cell(data.numItems, 1);
        samp.njU = cell(data.numItems, 1);
        samp.tjU = cell(data.numItems, 1); % Dish assignments for each item's all examples
        samp.tU = uint32(repmat(0, 1, data.numExamples));
        samp.kU = uint32(repmat(0, 1, data.numExamples));
        
        % Sample topics for all examples
        samp = randomInitializeCRF(samp, data);
        samp = batchUpdate(samp, data, model, topicModel);
        samp = sampleCRFBias(samp, model, true, iscollapsed);
        samp = sampleCRFBias(samp, model, false, iscollapsed);
        
        % To be compatible with other functions, need to keep zU, zM
        samp.zU = samp.kU'; 
        samp.zM = samp.kM';
end


% TODO sample Z's from scratch, remove above z random initialization


sgdFactorVectors(data, model, samp, 10);
%samp = orderfields(samp);

% -----------------------------END OF CODE-------------------------------
