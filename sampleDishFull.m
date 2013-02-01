function k_new = sampleDishFull(examps,resids,mC,muC,muD,kU,nC, nD, model,isItemTopic)

if isItemTopic
    betaM = model.betaM; c0 = model.c0; 
else
    betaM = model.betaU; c0 = model.d0;
end

% Assemble multinomial probability vector for each existing dish and a new dish
KM = length(mC);
logmult = zeros(1,KM+1);
for kk = 1:KM
    for ee = examps
        residC = resids(ee) - muD(kU(ee));
        sigmaStarSqd = getSigmaStarSqd(kk, kU(ee), nC, nD, model);
        logmult(kk) = logmult(kk) - (residC - double(muC(kk))) ^ 2 / sigmaStarSqd / 2 - log(sigmaStarSqd) / 2;
    end
end
for ee = examps
    residC = resids(ee) - muD(kU(ee));
    sigmaDSqd = getSigmaDSqd(kU(ee), nD, model);
    % TODO add sigmaC
    logmult(KM+1) = logmult(KM+1) - (residC - c0) ^ 2 / (model.sigmaSqd + sigmaDSqd) / 2 - log(sigmaDSqd + model.sigmaSqd) / 2;
end
logmult = logmult - mean(logmult);
logmult(logmult>30) = 30; % Make sure it doesn't overflow.

mult = [mC, betaM] .* exp(logmult);
mult_norm = mult / sum(mult); % Normalize;

% Sample the new dish, and update related fields
k_new = find(mnrnd(1, mult_norm));

if length(k_new) > 1
    display(mC);display(logmult);display(mult);
end