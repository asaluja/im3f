function samp = randomInitializeCRF(samp, data)

% Assign all examples to first table, and first table randomly to existing
% dishes

for uu = 1:data.numUsers
    numExamps = length(data.exampsByUser{uu});
    % Just 1 table currently
    samp.kuM{uu} = uint32([randi(samp.KM)]);
    samp.nuM{uu} = uint32([numExamps]); 
    samp.tuM{uu} = uint32(repmat(1,1,numExamps)); % All examps go to table 1
end


for jj = 1:data.numItems
    numExamps = length(data.exampsByItem{jj});
    % Just 1 table currently
    samp.kjU{jj} = uint32([randi(samp.KU)]);
    samp.njU{jj} = uint32([numExamps]); 
    samp.tjU{jj} = uint32(repmat(1,1,numExamps)); % All examps go to table 1
end