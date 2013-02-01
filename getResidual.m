function samp = getResidual(data,model,samp)
% Every time muC, muD value changes, residual changes

tResid = tic;
resids = zeros(1, data.numExamples);

users = data.users;
items = data.items;
vals = data.vals;

%for ee = 1:data.numExamples
%    uu = users(ee);
%    jj = items(ee);
%    resids(ee) = vals(ee) - model.chi0 - samp.a(:,uu)' * samp.b(:,jj);    
%end

%ds = samp.muD(samp.kU); cs = samp.muC(samp.kM);    

%samp.resids = resids;
samp.resids = getResidualMex(users, items, vals, samp); 
%samp.residCs = resids - ds; 
%samp.residDs = resids - cs;

fprintf('\tgetResidual------%g\n',toc(tResid));