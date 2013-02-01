function [preds] = m3f_tib_predictToney(users, items, samp, zU, zM, topicModel)
%(data.users, data.items, samp, samp.zU, samp.zM, [true, true, true], topicModelIndex(topicModel));

if isempty(zU) % Cache expected biases
    uniq_users = unique(users)';
    uniq_items = unique(items)';
    expected_c = zeros(1, max(uniq_users));
    expected_d = zeros(1, max(uniq_items));
    
    switch topicModel
        case 'secrp'
            for uu = uniq_users
                expected_c(uu) = samp.nC(:,uu)' * samp.muC(:,uu) / (sum(samp.nC(:,uu))+0.0000001);
            end
            for jj = uniq_items
                expected_d(jj) = samp.nD(:,jj)' * samp.muD(:,jj) / (sum(samp.nD(:,jj))+0.0000001);
            end            
        case 'crf'
            for uu = uniq_users
                c = 0;
                for tt = 1:length(samp.nuM{uu})
                    if samp.kuM{uu}(tt) == 0
                        continue
                    end
                    c = c + double(samp.nuM{uu}(tt)) * samp.muC(samp.kuM{uu}(tt));
                end
                expected_c(uu) = c / (sum(samp.nuM{uu})+0.0000001);
            end
            for jj = uniq_items
                c = 0;
                for tt = 1:length(samp.njU{jj})
                    if samp.kjU{jj}(tt) == 0
                        continue
                    end
                    c = c + double(samp.njU{jj}(tt)) * samp.muD(samp.kjU{jj}(tt));
                end
                expected_d(jj) = c / (sum(samp.njU{jj})+0.0000001);
            end
    end
end

N = length(users);
preds = ones(N,1) * samp.chi;
for ee = 1:N
    uu = users(ee); jj = items(ee);
    preds(ee) = preds(ee) + samp.a(:,uu)' * samp.b(:,jj);
    if isempty(zU) % Test data
        c = expected_c(uu); d = expected_d(jj);
    else % Train data
        switch topicModel
            case 'secrp'
                c = samp.muC(zM(ee),uu); d = samp.muD(zU(ee),jj);
            case 'crf'
                c = samp.muC(zM(ee)); d = samp.muD(zU(ee));
        end
    end
    preds(ee) = preds(ee) + c + d;
end