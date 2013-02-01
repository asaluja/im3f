function samp = sampleTopicsCRF(data,model,samp,iscollapsed)

samp = getResidual(data,model,samp);

samp = sampleCRFTablesRealRealTime(data, model, samp, true, iscollapsed);
samp = sampleCRFBias(samp, model, true, iscollapsed);

samp = sampleCRFTablesRealRealTime(data, model, samp, false, iscollapsed);
samp = sampleCRFBias(samp, model, false, iscollapsed);


samp = sampleCRFDishsRealTime(samp, data, model, true, iscollapsed);
samp = sampleCRFBias(samp, model, true, iscollapsed);

samp = sampleCRFDishsRealTime(samp, data, model, false, iscollapsed);
samp = sampleCRFBias(samp, model, false, iscollapsed);
samp = sampleCRFBias(samp, model, true, iscollapsed);

%samp = sampleCRFBias(samp,model);




















