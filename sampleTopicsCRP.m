function [samp] = sampleTopicsCRP(data,model,samp,topicModel,isItemTopic)
%function [muC,nC,muD,nD,KM,KU,zM,zU] = sampleTopicsCRP(data,model,samp)


if isItemTopic == true
    muC = samp.muC;
    muD = samp.muD;
    nC = samp.nC;
    KM = samp.KM;
    zM = samp.zM;    
    zU = samp.zU;
    users = data.users;
    items = data.items;
    numUsers = data.numUsers;
    c0 = model.c0;
    a = samp.a;
    b = samp.b;
else % Switch roles
    muC = samp.muD;
    muD = samp.muC;
    nC = samp.nD;
    KM = samp.KU;
    zM = samp.zU;
    zU = samp.zM;
    users = data.items;
    items = data.users;
    numUsers = data.numItems;
    c0 = model.d0;
    a = samp.b;
    b = samp.a;
end

% Toney - need to verify this
% Iterate all (u,j) pair
% Function written from perspective of updating item topics (muC, nC, zM)
for ee = 1:length(data.vals)
    uu = users(ee);
    jj = items(ee); % item id
    ii = zM(ee); % item topic
    ii_u = zU(ee); % user topic
    
    resid = data.vals(ee) - model.chi0 - a(:,uu)' * b(:,jj);
    
    switch topicModel
        case 'secrp'
            % Sample item topic
            residC = resid - muD(ii_u,jj);
            % Assemble CRP multinomial
            mult = zeros(1, KM + 1);
            % Likelihood for new topic
            l_new = model.gammaM * exp(- residC * residC * model.invsigmaSqd / 2);
            %mult(KM + 1) = l_new;
            new_topic_index = KM + 1; % Initially assume all existing topic have items
            for iii = 1:KM
                % Calculate data likelihood
                if nC(iii,uu) > 0 % Has data points in the topic
                    r = residC - muC(iii,uu);
                    l = double(nC(iii,uu)) * exp(- r * r * model.invsigmaSqd / 2);
                else
                    %l = l_new; % Toney - need to decide this
                    l = 0;
                    new_topic_index = iii; % Once finds an existing topic that have no items, use that to store new topic items
                end
                mult(iii) = l;
            end
            mult(new_topic_index) = l_new;
            mult_norm = mult / sum(mult); % normalize
            % Sample from the multinomial
            zMe = find(mnrnd(1,mult_norm));
            if length(zMe) > 1
                zMe = randi(KM+1);
            end
            % Update mu,n matrix if topic changed, otherwise do nothing
            if zMe ~= ii
                if zMe > KM % Needs to assign new topic
                    KM = KM + 1;
                    muC = [muC; repmat(c0, 1, numUsers)]; %#ok<*AGROW>
                    nC = [nC; repmat(c0, 1, numUsers)]; %#ok<*AGROW>
                end
                [new_muC, new_nC] = updateMuC(muC, nC, uu, zMe, residC, model.invsigmaSqd0, model.invsigmaSqd, true);
                muC(zMe,uu) = new_muC; nC(zMe,uu) = new_nC; % Add to new topic
                [new_muC, new_nC] = updateMuC(muC, nC, uu, ii, residC, model.invsigmaSqd0, model.invsigmaSqd, false);
                muC(ii,uu) = new_muC; nC(ii,uu) = new_nC; % Remove from original topic
            end
            % Assign new topic
            zM(ee) = zMe;
        case 'crf'
            error('Not implemented yet');
    end
end

if isItemTopic == true
    samp.muC = muC;
    samp.nC = nC;
    samp.KM = KM;
    samp.zM = zM;
else
    samp.muD = muC;
    samp.nD = nC;
    samp.KU = KM;
    samp.zU = zM;
end
