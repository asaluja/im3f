function samp = update_tM_kM_tU_kU(samp, data)

% Updates tM, kM matrices according to current kuM, tuM
% Similarly updates tU, kU as well.

for uu = 1:data.numUsers
    examps = data.exampsByUser{uu};
    tuM = samp.tuM{uu};
    kuM = samp.kuM{uu};
    for ee_i = 1 : length(examps)
        ee = examps(ee_i);
        samp.tM(ee) = tuM(ee_i);
        samp.kM(ee) = kuM(tuM(ee_i));
    end
end

for jj = 1:data.numItems
    examps = data.exampsByItem{jj};
    tjU = samp.tjU{jj};
    kjU = samp.kjU{jj};
    for ee_i = 1 : length(examps)
        ee = examps(ee_i);
        samp.tU(ee) = tjU(ee_i);
        samp.kU(ee) = kjU(tjU(ee_i));
    end
end