function samp = sampleTopicsCRF(data,model,samp,iscollapsed)

samp = getResidual(data,model,samp);
tStart = tic; 
%samp = sampleCRFTablesRealRealTime(data, model, samp, true, iscollapsed);
[samp.kuM, samp.nuM, samp.tuM, samp.mC, samp.nC, samp.muC, samp.tM, ...
samp.kM] = sampleCRFTablesMex(data, model, samp, [1, iscollapsed]);   
fprintf('\tsampleCRFTablesRealRealTime------%g\n',toc(tStart));
samp = sampleCRFBias(samp, model, true, iscollapsed);

tStart = tic; 
%samp = sampleCRFTablesRealRealTime(data, model, samp, false, iscollapsed); 
[samp.kjU, samp.njU, samp.tjU, samp.mD, samp.nD, samp.muD, samp.tU, ...
samp.kU] = sampleCRFTablesMex(data, model, samp, [0, iscollapsed]); 
fprintf('\tsampleCRFTablesRealRealTime------%g\n',toc(tStart));
samp = sampleCRFBias(samp, model, false, iscollapsed);

tStart = tic; 
%samp = sampleCRFDishsRealTime(samp, data, model, true,iscollapsed);
[samp.kuM, samp.mC, samp.nC, samp.muC, samp.kM] = sampleCRFDishsMex(data, ...
                                                  model, samp, [1, iscollapsed]); 
fprintf('\tsampleCRFDishesRealTime------%g\n', toc(tStart)); 
samp = sampleCRFBias(samp, model, true, iscollapsed);

tStart = tic; 
%samp = sampleCRFDishsRealTime(samp, data, model, false, iscollapsed);
[samp.kjU, samp.mD, samp.nD, samp.muD, samp.kU] = sampleCRFDishsMex(data, ...
                                                  model, samp, [0, iscollapsed]); 
fprintf('\tsampleCRFDishesRealTime------%g\n', toc(tStart)); 
samp = sampleCRFBias(samp, model, false, iscollapsed);
samp = sampleCRFBias(samp, model, true, iscollapsed);

%samp = sampleCRFBias(samp,model);




















