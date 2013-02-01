function l = m3f_tib_getL(model,data,samp)

N = length(data.vals);
l = zeros(N,1);

for ee = 1:N
    jj = data.items(ee); % item id
    uu = data.users(ee); % user id
    ii = samp.zM(ee); % item topic
    ii_u = samp.zU(ee); % user topic
         
    resid = data.vals(ee) - model.chi0 - samp.a(:,uu)' * samp.b(:,jj) - samp.muC(ii,uu) - samp.muD
    residC = resid - samp.muD(ii_u,jj);
    [new_muC, new_nC] = updateMuC(samp.muC, samp.nC, uu, ii, residC, model.invsigmaSqd0, model.invsigmaSqd, true);
    samp.muC(ii,uu) = new_muC; samp.nC(ii,uu) = new_nC;
    
    % Update muD
    residD = resid - samp.muC(ii,uu);
    % Use same update function, but change roles
    [new_muD, new_nD] = updateMuC(samp.muD, samp.nD, jj, ii_u, residD, model.invsigmaSqd0, model.invsigmaSqd, true);
    samp.muD(ii_u,jj) = new_muD; samp.nD(ii_u,jj) = new_nD;   
end