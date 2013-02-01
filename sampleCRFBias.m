function samp = sampleCRFBias(samp, model, isItemTopic, iscollapsed)

switch iscollapsed
    case 'collapsed'        
        if isItemTopic
            samp.c = samp.muC;
        else
            samp.d = samp.muD;
        end
    case 'noncollapsed'
        if isItemTopic
            samp.c = randn(size(samp.muC)) ./ (samp.nC * model.invsigmaSqd + model.invsigmaSqd0) + samp.muC;
        else
            samp.d = randn(size(samp.muD)) ./ (samp.nD * model.invsigmaSqd + model.invsigmaSqd0) + samp.muD;
        end
        
    otherwise
        error('Unknown collapsed option');
end