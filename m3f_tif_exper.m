function [err] = m3f_tif_exper(experName, dataName, splitNums, initMode, ...
                                 seed, numFacs, KU, KM, numTopicFacs)
%M3F_TIF_EXPER Run m3f_tif Gibbs sampling experiments.
%
% Usage:
%    [err] = m3f_tif_exper(experName, dataName, splitNums, initMode, seed, 
%            seed, numFacs, KU, KM, numTopicFacs)
%
% Inputs:
%    experName - Name of experiment
%    dataName - Name of dataset
%    splitNums - Indices of dataset test-train splits
%    initMode - Sample initialization mode (see m3f_tif_initSamp)
%    seed - Seed for random number generators
%    numFacs - Number of static latent factors
%    KU - Number of user topics
%    KM - Number of item topics
%    numTopicFacs - Number of topic-indexed factors
%
% Outputs:
%    err - Error on training and test sets following each Gibbs sampling
%          round

% References:
%    Mackey, Weiss, and Jordan, "Mixed Membership Matrix Factorization,"
%    International Conference on Machine Learning, 2010.

% -----------------------------------------------------------------------     
%
% Last revision: 12-July-2010
%
% Authors: Lester Mackey and David Weiss
% License: MIT License
%
% Copyright (c) 2010 Lester Mackey & David Weiss
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------

% -----------------------------BEGIN CODE--------------------------------

%% For each train-test split
for s = splitNums
   %% Seed random number generators
   seedRand(seed);
   
   %% Load Train and Test Datasets
   useHoldOut = false; % Use hold out in place of test data?
   modelID = sprintf('-numFacs%d_KU%d_KM%d_numTopicFacs%d', numFacs, ...
                     KU, KM, numTopicFacs);
   fprintf('Loading data set %s with split %d, model %s, and init %d\n', ...
           dataName, s, modelID, initMode);
   [data, testData] = loadDyadicData(dataName, s, useHoldOut);
   
   %% Initialize model
   model = m3f_tif_initModel(data.numUsers, data.numItems, numFacs, ...
                               KU, KM, numTopicFacs);
   model.chi0 = mean(data.vals);
   prefix = 'm3f_tif';
   save(sprintf([experName, '/models/',prefix,'_%s_split%d_model%s'], ...
                dataName, s, modelID), 'model');
   
   %% Choose initial sample
   samp = m3f_tif_initSamp(initMode, model, data, testData);

   %% Create Gibbs sampling options structure
   % Number of sampling rounds
   opts.T = 50;
   % Number of burnin rounds
   opts.burnin = 0;
   % Log file string
   opts.logStr = sprintf([experName,'/log/',prefix,'_%s_split%d_model%s_init%d.log'], ...
                         dataName, s, modelID, initMode);
   % Sample output file format string
   opts.formatStr = sprintf([experName, '/samples/',prefix,'_%s_split%d_model%s_init%d_sample%%d'], ...
                            dataName, s, modelID, initMode);
   % Disable saving of samples to disk after each sampling round?
   % If true, only final sample will be saved to disk.
   opts.disableSaving = true;
   
   %% Perform gibbs sampling on model
   err = m3f_tif_gibbs(data, model, samp, opts, testData);
   clear data testData;
end

% -----------------------------END OF CODE-------------------------------
