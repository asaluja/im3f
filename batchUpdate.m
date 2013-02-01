function samp = batchUpdate(samp, data, model, topicModel)

switch topicModel
    case 'secrp'
        for ee = 1:data.numExamples
            jj = data.items(ee); % item id
            uu = data.users(ee); % user id
            ii = samp.zM(ee); % item topic
            ii_u = samp.zU(ee); % user topic
            
            resid = data.vals(ee) - model.chi0 - samp.a(:,uu)' * samp.b(:,jj);
            
            % Update muC, muD
            residC = resid - samp.muD(ii_u,jj); residD = resid - samp.muC(ii,uu);
            [new_muC, new_nC] = updateMuC(samp.muC, samp.nC, uu, ii, residC, model.invsigmaSqd0, model.invsigmaSqd, true);
            samp.muC(ii,uu) = new_muC; samp.nC(ii,uu) = new_nC;
            [new_muD, new_nD] = updateMuC(samp.muD, samp.nD, jj, ii_u, residD, model.invsigmaSqd0, model.invsigmaSqd, true);
            samp.muD(ii_u,jj) = new_muD; samp.nD(ii_u,jj) = new_nD;
        end
    case 'crf'        
        samp = update_tM_kM_tU_kU(samp, data);
        
        % Clear existing result
        KM = length(samp.muC); samp.muC = repmat(0, 1, KM); samp.mC = repmat(0, 1, KM); samp.nC = repmat(0, 1, KM); 
        KU = length(samp.muD); samp.muD = repmat(0, 1, KU); samp.mD = repmat(0, 1, KU); samp.nD = repmat(0, 1, KU); 
        
        % Update m, n
        for uu = 1:data.numUsers
            kuM = samp.kuM{uu};
            for ii = 1:length(kuM) 
                if samp.nuM{uu}(ii) ~= 0 % Don't count empty table
                    if kuM(ii) > length(samp.mC)
                        samp.mC(kuM(ii)) = 0;
                        samp.nC(kuM(ii)) = 0;
                        samp.muC(kuM(ii)) = model.c0 * model.sigmaSqd / model.sigmaSqd0; % Initialize mu for the new dish
                    end
                    samp.mC(kuM(ii)) = samp.mC(kuM(ii)) + 1;
                    samp.nC(kuM(ii)) = samp.nC(kuM(ii)) + samp.nuM{uu}(ii);     
                end
            end
        end
        for jj = 1:data.numItems
            kjU = samp.kjU{jj};
            for ii = 1:length(kjU)
                if samp.njU{jj}(ii) ~= 0 % Don't count empty table
                    if kjU(ii) > length(samp.mD)
                        samp.mD(kjU(ii)) = 0;
                        samp.nD(kjU(ii)) = 0;
                        samp.muD(kjU(ii)) = model.d0 * model.sigmaSqd / model.sigmaSqd0; % Initialize mu for the new dish
                    end
                    samp.mD(kjU(ii)) = samp.mD(kjU(ii)) + 1;
                    samp.nD(kjU(ii)) = samp.nD(kjU(ii)) + samp.njU{jj}(ii);
                end
            end
        end
        
        samp = getResidual(data,model,samp);
        
        % Update mu
        for ee = 1 : data.numExamples
            kU = samp.kU(ee); kM = samp.kM(ee);             
            samp.muC(kM) = samp.muC(kM) + samp.resids(ee) / 2;
            samp.muD(kU) = samp.muD(kU) + samp.resids(ee) / 2; 
        end
        
        % mu and n should be equal length.
        % Adding a small amount so when nC is 0, it doesn't get to NaN.
        samp.muC = samp.muC ./ (samp.nC + model.sigmaSqd / model.sigmaSqd0);
        samp.muD = samp.muD ./ (samp.nD + model.sigmaSqd / model.sigmaSqd0);   
        
        
end










