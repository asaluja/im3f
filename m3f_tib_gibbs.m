function [err] = m3f_tib_gibbs(data, model, samp, topicModel, iscollapsed, opts, testData)
%M3F_TIB_GIBBS Gibbs sampler for TIB model.

fprintf('Running m3f_tib_gibbs\n');

fprintf('Number of rounds: %d (burnin %d)\n', opts.T, opts.burnin);

%% Extract size information from model
numUsers = model.numUsers;
numItems = model.numItems;

%% Auxiliary variables
eyeSizeNumFacs = eye(size(model.W0));
W0inv = model.W0\eyeSizeNumFacs;

%% Open stream for outputting log data and err data
logStrm = fopen(opts.logStr,'w');
errStrm = fopen(opts.errStr,'w');

trainPreds = zeros(size(data.vals));
testPreds = zeros(size(testData.vals));

%% Perform Gibbs sampling
for t = 1:opts.T
    fprintf('Collecting sample %d\n', t);
    tStart = tic; % Measure time elapsed

    %% Sample hyperparameters
    fprintf('Sampling hyperparameters...\n');
    [samp.LambdaU, samp.muU] = sampleHyperParams(model, samp.a, W0inv, numUsers,eyeSizeNumFacs);
    [samp.LambdaM, samp.muM] = sampleHyperParams(model, samp.b, W0inv, numItems, eyeSizeNumFacs);
    
    %% Sample user and item topics for each rating and topic parameters
    fprintf('Sampling topics and topic parameters...\n');
    tTopic = tic;
    switch topicModel
        case 'secrp'
            samp = sampleTopicsCRP(data,model,samp,topicModel,true);
            samp = sampleTopicsCRP(data,model,samp,topicModel,false);            
            samp.c = samp.muC; samp.d = samp.muD; % Toney - use c = muC, d=muD
        case 'crf'
            samp = sampleTopicsCRF(data,model,samp,iscollapsed);
            samp.zU = samp.kU'; samp.zM = samp.kM'; % Legacy variables
    end
    tElap = toc(tTopic); fprintf('Sampling topics and topic parameters...%f seconds\n',tElap);
    
    %% Sample factor vectors
    fprintf('Sampling factor vectors...\n');
    tic;
    switch topicModel
        case 'secrp'
            m3f_tib_sampleFactorVectors(data, model, samp, samp.zU, samp.zM, false);
        case 'crf'
            m3f_tib_sampleFactorVectorsCRF(data, model, samp, samp.kU', samp.kM', false);
    end
    toc;

    %% Form ratings predictions
    if (t > opts.burnin)
        trainPreds = trainPreds + m3f_tib_predictToney(data.users, data.items, samp, samp.zU, samp.zM, topicModel);
        testPreds = testPreds +  m3f_tib_predictToney(testData.users, testData.items, samp, [], [], topicModel);
    end    
    % Evaluate ratings predictions
    trainRMSE = evalPreds(data.vals, trainPreds/(t-opts.burnin), 'rmse');
    trainMAE = evalPreds(data.vals, trainPreds/(t-opts.burnin), 'mae');
    testRMSE = evalPreds(testData.vals, testPreds/(t-opts.burnin), 'rmse');
    testMAE = evalPreds(testData.vals, testPreds/(t-opts.burnin), 'mae');
    fprintf(logStrm, ['Round %d Eval:\n\tTrain RMSE = %g, Train MAE = %g\n\t','Test RMSE = %g, Test MAE = %g\n'], t, trainRMSE, trainMAE, testRMSE, testMAE);
    err(t) = struct('trainRMSE',trainRMSE,'trainMAE',trainMAE,'testRMSE', testRMSE,'testMAE',testMAE); %#ok<AGROW>
    display(err(t));
    
    %% Diagnose convergence
    
    luNorm = norm(samp.LambdaU,'fro');
    lmNorm = norm(samp.LambdaM,'fro');
    aNorm = norm(samp.a,'fro');
    bNorm = norm(samp.b,'fro');
    cNorm = norm(samp.c,'fro');
    dNorm = norm(samp.d,'fro');
    switch topicModel
        case 'secrp'
            zUcounts = accumarrayMex(samp.zU, 1, [samp.KU,1]);
            zMcounts = accumarrayMex(samp.zM, 1, [samp.KM,1]);
        case 'crf'
            zUcounts = samp.nD;
            zMcounts = samp.nC;
    end
    % Count number of topics (with non-zero rating count)
    ZU = length(find(zUcounts)); ZM = length(find(zMcounts));
    
    % Count tables in each topic
    tableStr = ''; muStr = '';
    if strcmp(topicModel,'crf')
        tableStr = ['\tmC = ', sprintf('%d ', samp.mC), '\n\tmD = ', sprintf('%d ', samp.mD)];
        muStr = ['\tmuC = ', sprintf('%g ', samp.muC), '\n\tmuD = ', sprintf('%g ', samp.muD),'\n'];
    end
    
    roundDiagnostics = sprintf(['Round %d Diagnostics:\n',...
                      '\tLambdaU norm = %g, LambdaM norm = %g\n',...
                      '\ta norm = %g, b norm = %g,\n',...
                      '\tc norm = %g, d norm = %g\n',muStr,...
                      '\tzU counts (%d non-zero)= ', sprintf('%d ', zUcounts),...
                      '\n\tzM counts (%d non-zero)= ', sprintf('%d ', zMcounts),...
                      '\n', tableStr,...                      
                      '\n'],...
            t, luNorm, lmNorm, aNorm, bNorm, cNorm, dNorm, ZU, ZM);
    fprintf(roundDiagnostics); fprintf(logStrm, roundDiagnostics);
    
    tElap = toc(tStart);
    fprintf('Finished round %d.\n----------------------------------------(%g seconds)\n\n', t, tElap);
    fprintf(logStrm, 'Elapsed time: %g seconds.\n', tElap);    
    fprintf(errStrm, '%d\t%g\t%g\t%g\t%g\t%d\t%d\n', t, trainRMSE, trainMAE, testRMSE, testMAE, ZU, ZM);

end

%% Close open streams
fclose(logStrm);
fclose(errStrm);

% -----------------------------END OF CODE-------------------------------
