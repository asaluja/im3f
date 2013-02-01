function [new_muC, new_nC] = updateMuC(muC, nC, u, i, resid, invsigmasqr0, invsigmasqr, isplus)

variance = invsigmasqr0 + nC(i,u) * invsigmasqr;
sum = muC(i,u) * variance;

changeMu = resid * invsigmasqr;
changeN = 1;
if isplus==false
    changeMu = -changeMu;
    changeN = -changeN;
end

variance = variance + changeN * invsigmasqr;
new_muC = (sum + changeMu) / variance;
new_nC = nC(i,u) + changeN;

%muC(i,u) = new_muC;
%nC(i,u) = new_nC;
    