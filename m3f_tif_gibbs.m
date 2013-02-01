function [err] = m3f_tif_gibbs(data, model, samp, opts, testData)
%M3F_TIF_GIBBS Gibbs sampler for TIF model.
%
% Usage:
%    [err] = m3f_tif_gibbs(data, model, samp, opts, testData)
%
% Inputs:
%    data - Dyadic data structure (see loadDyadicData)
%    model - m3f_tif structure (see m3f_tif_initModel)
%    samp - Sample used to initialize the Markov chain (see m3f_tif_initSamp)
%    opts - Options structure:
%           opts.T => Number of sampling rounds (including initial sample)
%           opts.burnin => Number of burnin rounds (including initial sample)
%           opts.logStr => File for outputting logging information
%           opts.formatStr => Template format string for files used to
%           save samples; should include %d placeholder for sample number
%           opts.disableSaving => Disable saving of samples to disk after
%           each sampling round?  If true, only final sample will be
%           saved to disk.
%    testData - Test dataset for reporting evaluation metrics
%
% Outputs:
%    err - Error on training and test sets following each Gibbs sampling
%          round

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

fprintf('Running m3f_tif_gibbs\n');

fprintf('Number of rounds: %d (burnin %d)\n', opts.T, opts.burnin);

%% Extract size information from model
numUsers = model.numUsers;
numItems = model.numItems;

%% Store example ids by user and by item
if ~isfield(data,'exampsByUser') || isempty(data.exampsByUser)
    data.exampsByUser = jaggedCell(data.users, numUsers);
end
if ~isfield(data,'exampsByItem') || isempty(data.exampsByItem)
    data.exampsByItem = jaggedCell(data.items, numItems);
end

%% Auxiliary variables
% numFacs x numFacs identity matrix
eyeSizeNumFacs = eye(size(model.W0));
% numTopicFacs x numTopicFacs identity matrix
eyeSizeNumTopicFacs = eye(size(model.WTilde0));
% Compute Inverse of W0, WTilde0
W0inv = model.W0\eyeSizeNumFacs;
WTilde0inv = model.WTilde0\eyeSizeNumTopicFacs;

%% Do not store topic vectors in samp
zU = samp.zU;
zM = samp.zM;
samp.zU = [];
samp.zM = [];

%% Open stream for outputting log data
logStrm = fopen(opts.logStr,'w');

%% Form ratings predictions using initial sample
tic;
trainPreds = m3f_tif_predictMex(data.users, data.items, samp, ...
                                  zU, zM, [true, true, true]);
testPreds = m3f_tif_predictMex(testData.users, testData.items, samp, ...
                                 [], [], [true, true, true]);
toc;
% Evaluate ratings predictions of initial sample
trainRMSE = evalPreds(data.vals, trainPreds, 'rmse');
trainMAE = evalPreds(data.vals, trainPreds, 'mae');
testRMSE = evalPreds(testData.vals, testPreds, 'rmse');
testMAE = evalPreds(testData.vals, testPreds, 'mae');
fprintf(logStrm, ['Round %d Eval:\n\tTrain RMSE = %g, Train MAE = %g\n\t',...
                  'Test RMSE = %g, Test MAE = %g\n'], 1, trainRMSE, trainMAE, ...
        testRMSE, testMAE);

%% Save current samp to file (don't save topics)
if ~opts.disableSaving
    save(sprintf(opts.formatStr, 1), 'samp', 'testRMSE', 'testMAE');
end
err = struct('trainRMSE',trainRMSE,'trainMAE',trainMAE,'testRMSE', ...
             testRMSE,'testMAE',testMAE);

%% Reset predictions if burnin enabled
if opts.burnin > 0
    trainPreds = zeros(size(trainPreds));
    testPreds = zeros(size(testPreds));
end

%% Perform Gibbs sampling
for t = 2:opts.T
    fprintf('Collecting sample %d\n', t);
    tStart = tic; % Measure time elapsed

    %% Sample hyperparameters
    fprintf('Sampling hyperparameters...\n');
    % Sample user hyperparameters
    [samp.LambdaU, samp.muU] = ...
        sampleHyperParams(model, samp.a, W0inv, numUsers,...
                          eyeSizeNumFacs);
    [samp.LambdaTildeU, samp.muTildeU] = ...
        m3f_tif_sampleTopicHyperParams(samp.c, WTilde0inv, model.nuTilde0, ...
                                    model.muTilde0, model.lambdaTilde0, ...
                                    model.KM*numUsers, eyeSizeNumTopicFacs);
    % Sample item hyperparameters
    [samp.LambdaM, samp.muM] = ...
        sampleHyperParams(model, samp.b, W0inv, numItems,...
                          eyeSizeNumFacs);
    [samp.LambdaTildeM, samp.muTildeM] = ...
        m3f_tif_sampleTopicHyperParams(samp.d, WTilde0inv, model.nuTilde0, ...
                                    model.muTilde0, model.lambdaTilde0, ...
                                    model.KU*numItems, eyeSizeNumTopicFacs);
    toc;

    %% Sample biases
    fprintf('Sampling biases...\n');
    tic;
    resids = data.vals - m3f_tif_predictMex(data.users, data.items, samp, ...
                                              zU, zM, [true, true, false]);
    m3f_tif_sampleBiases(data, model, samp, resids);
    toc;

    %% Sample user and item topics for each rating and topic parameters
    fprintf('Sampling topics and topic parameters...\n');
    tic;
    resids = data.vals - m3f_tif_predictMex(data.users, data.items, samp, ...
                                              zU, zM, [true, false, true]);
    m3f_tif_sampleTopics(data, model, samp, zU, zM, resids);
    sampleTopicParams(data, model, samp, zU, zM);
    toc;

    %% Sample topic factors
    fprintf('Sampling topic factor vectors...\n');
    tic;
    m3f_tif_sampleTopicFactorVectors(data, model, samp, zU, zM, resids);
    toc;

    %% Sample factor vectors
    fprintf('Sampling factor vectors...\n');
    tic;
    resids = data.vals - m3f_tif_predictMex(data.users, data.items, samp, ...
                                              zU, zM, [false, true, true]);
    m3f_tif_sampleFactorVectors(data, model, samp, resids);
    toc;

    %% Form ratings predictions
    tic;
    if (t > opts.burnin)
       trainPreds = trainPreds + ...
           m3f_tif_predictMex(data.users, data.items, samp, zU, zM, [true, true, true]);
       testPreds = testPreds + m3f_tif_predictMex(testData.users, testData.items, samp,...
                                                    [], [], [true, true, true]);
    end
    toc;
    trainRMSE = evalPreds(data.vals, trainPreds/(t-opts.burnin), 'rmse');
    trainMAE = evalPreds(data.vals, trainPreds/(t-opts.burnin), 'mae');
    testRMSE = evalPreds(testData.vals, testPreds/(t-opts.burnin), 'rmse');
    testMAE = evalPreds(testData.vals, testPreds/(t-opts.burnin), 'mae');
    fprintf(logStrm, ['Round %d Eval:\n\tTrain RMSE = %g, Train MAE = %g\n\t',...
                      'Test RMSE = %g, Test MAE = %g\n'], t, trainRMSE, ...
            trainMAE, testRMSE, testMAE);
    err(t) = struct('trainRMSE',trainRMSE,'trainMAE',trainMAE,'testRMSE', ...
                 testRMSE,'testMAE',testMAE);
    display(err(t));

    %% Save current sample to file (without topics)
    if ~opts.disableSaving || t == opts.T 
       if model.KU > 0
          zUcounts = accumarrayMex([zU, data.users], 1, [model.KU,model.numUsers]);
       else
          zUcounts = 0;
       end
       if model.KM > 0
          zMcounts = accumarrayMex([zM, data.items], 1, [model.KM,model.numItems]);
       else
          zMcounts = 0;
       end
       save(sprintf(opts.formatStr, t), 'samp', 'trainRMSE', 'trainMAE', ...
            'testRMSE', 'testMAE', 'zUcounts', 'zMcounts');
    end

    tElap = toc(tStart);
    fprintf('Finished round %d.\n', t);
    fprintf(logStrm, 'Elapsed time: %g seconds.\n', tElap);

    %% Diagnose convergence
    luNorm = norm(samp.LambdaU,'fro');
    lmNorm = norm(samp.LambdaM,'fro');
    ltuNorm = norm(samp.LambdaTildeU,'fro');
    ltmNorm = norm(samp.LambdaTildeM,'fro');
    aNorm = norm(samp.a,'fro');
    bNorm = norm(samp.b,'fro');
    for i = 1:model.KM
       cNorm(i) = norm(samp.c(:,:,i),'fro');
    end
    for i = 1:model.KU
       dNorm(i) = norm(samp.d(:,:,i),'fro');
    end
    if model.KU > 0
        zUcounts = accumarrayMex(zU, 1, [model.KU,1]);
    else
        zUcounts = 0;
    end
    if model.KM > 0
        zMcounts = accumarrayMex(zM, 1, [model.KM,1]);
    else
        zMcounts = 0;
    end
    fprintf(logStrm, ['Round %d Diagnostics:\n',...
                      '\tLambdaU norm = %g, LambdaM norm = %g\n',...
                      '\tLambdaTildeU norm = %g, LambdaTildeM norm = %g\n',...
                      '\ta norm = %g, b norm = %g,\n',...
                      '\tc norm = ',...
                      sprintf('%d ', cNorm),...
                      '\n\td norm = ',...
                      sprintf('%d ', dNorm),...
                      '\n\tzU counts = ',...
                      sprintf('%d ', zUcounts),...
                      '\n\tzM counts = ',...
                      sprintf('%d ', zMcounts),...
                      '\n'],...
            t, luNorm, lmNorm, ltuNorm, ltmNorm, aNorm, bNorm);
end

%% Close open streams
fclose(logStrm);

% -----------------------------END OF CODE-------------------------------