function [err] = m3f_tib_exper(experDir, experName, dataName, splitNames, topicModel, iscollapsed, initMode, seed, numFacs, KU, KM, gamma, beta, T)
%M3F_TIB_EXPER Run m3f_tib Gibbs sampling experiments.
%
% Usage:
%    [err] = m3f_tib_exper(experName, dataName, splitNames, initMode,
%            seed, numFacs, KU, KM)
%
% Inputs:
%    experDir - Directory to store & load experiment result
%    experName - Name of experiment
%    dataName - Name of dataset
%    splitNames - Cell array of names of dataset train-test splits
%    topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}
%    initMode - Sample initialization mode (see m3f_tib_initSamp)
%    seed - Seed for random number generators
%    numFacs - Number of static latent factors
%    KU - Number of user topics
%    KM - Number of item topics
%    T - Number of sampling rounds
%
% Outputs:
%    err - Error on training and test sets following each Gibbs sampling
%          round

% -----------------------------BEGIN CODE--------------------------------


%% Create log directories
for ss = {'','models','samples','log','errs'}
    dir = [experDir, '/', experName, '/', ss{1}];
    if exist(dir,'dir') == false
        mkdir(dir); % Create log directories
    end
end

expr = tic;
%% For each train-test split
for s = splitNames
   s = s{1}; %#ok<FXSET> % splitNames is cell array
   %% Seed random number generators
   seedRand(seed);
   
   %% Load Train and Test Datasets
   useHoldOut = false; % Use hold out in place of test data?
   modelID = sprintf('-%s_%s_numFacs%d_KU%d_KM%d_gamma%g_beta%g_T%g', topicModel, iscollapsed, numFacs, KU, KM, gamma, beta,T);
   fprintf('Loading data set %s with split %s, model %s, and init %d\n', dataName, s, modelID, initMode);
   %[data, testData] = loadDyadicDataToney(dataName, s, useHoldOut);
   [data, testData] = loadDyadicDataAvneesh(dataName, s, useHoldOut);
   
   %% Initialize model
   model = m3f_tib_initModel(data.numUsers, data.numItems, numFacs, KU, KM, gamma, beta);
   model.chi0 = mean(data.vals);
   save(sprintf([experDir, '/', experName,'/models/',experName,'_%s_split%s_model%s'], dataName, s, modelID), 'model');
   
   %% Choose initial sample
   %samp = m3f_tib_initSamp(topicModel, iscollapsed, model, data);   
   samp = m3f_tib_initSamp_v2(initMode, iscollapsed, model, data);   
   
   %% Create Gibbs sampling options structure   
   opts.T = T; % Number of sampling rounds (including initial sample)   
   opts.burnin = 0; % Number of burnin rounds
   opts.logStr = sprintf([experDir, '/', experName,'/log/',experName,'_%s_split%s_model%s_init%d.log'], dataName, s, modelID, initMode);
   opts.formatStr = sprintf([experDir, '/', experName,'/samples/',experName,'_%s_split%s_model%s_init%d_sample%%d'], dataName, s, modelID, initMode);
   opts.errStr = sprintf([experDir, '/', experName,'/errs/',experName,'_%s_split%s_model%s_init%d'], dataName, s, modelID, initMode);
   
   % Disable saving of samples to disk after each sampling round?
   % If true, only final sample will be saved to disk.
   opts.disableSaving = true;
   
   %% Perform gibbs sampling on model
   err = m3f_tib_gibbs(data, model, samp, topicModel, iscollapsed, opts, testData);
   
   clear data testData;
end

toc(expr);
% -----------------------------END OF CODE-------------------------------
