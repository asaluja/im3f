function [sample] = m3f_tib_rnd(model, uIDs, mIDs, sampleRatings)
%M3F_TIB_RND Sample from m3f_tib prior.
%
% Usage:
%    [sample] = m3f_tib_rnd(model)
%    [sample] = m3f_tib_rnd(model, uIDs, mIDs)
%    [sample] = m3f_tib_rnd(model, uIDs, mIDs, sampleRatings)
%
% Inputs:
%    model - m3f_tib structure (see m3f_tib_initModel)
%    uIDs - Array of user ids. If present, user and item topics will be
%    sampled for each user id, item id pair.
%    mIDs - Array of item ids. If present, user and item topics will be
%    sampled for each user id, item id pair.
%    sampleRatings - Logical. If true, ratings will be sampled for each
%    user id, item id pair.
%
% Outputs:
%    sample - Populated sample structure, drawn from model prior (see
%    m3f_tib_initSamp)

% -----------------------------------------------------------------------     
%
% Last revision: 2-July-2010
%
% Authors: Lester Mackey and David Weiss
% License: MIT License
%
% Copyright (c) 2010 Lester Mackey & David Weiss
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------

% -----------------------------BEGIN CODE--------------------------------

%% Helper functions
% Function for sampling topic parameter vector
function logtheta = logtheta_rnd(alpha, KU, numUsers)
if KU > 1
   logtheta = log(dirichlet_sample(repmat(alpha/KU, KU, 1), numUsers));
elseif KU > 0
   logtheta = zeros(1, numUsers);
else
   logtheta = [];
end
end

% Helper variables
eyeNumFacs = eye(size(model.W0));
numUsers = model.numUsers;
numItems = model.numItems;

% Set chi = chi0
chi = model.chi0; 

%% Sample all remaining model parameters
if size(eyeNumFacs, 1) > 0
   % Sample hyperparameters
   if model.nu0 > 0
      LambdaU = wishrnd(model.W0, model.nu0);
      LambdaM = wishrnd(model.W0, model.nu0);
   else
      LambdaU = model.W0*model.nu0;
      LambdaM = model.W0*model.nu0;
   end
   % Sample a and b static factor vectors
   LambdaUinv = LambdaU\eyeNumFacs;
   LambdaMinv = LambdaM\eyeNumFacs;
   muU = randnorm(1,model.mu0,[], LambdaUinv/model.lambda0);
   muM = randnorm(1,model.mu0,[], LambdaMinv/model.lambda0);
   a = randnorm(numUsers,muU,[], LambdaUinv);
   b = randnorm(numItems, muM,[], LambdaMinv);
else
   LambdaU = []; LambdaM = [];
   muU = []; muM = [];
   a = []; b = [];
end
% Sample c and d offsets
c = normrnd(model.c0, model.sigmaSqd0, model.KM, numUsers);
d = normrnd(model.d0, model.sigmaSqd0, model.KU, numItems);
          
% Sample logtheta topic parameters
logthetaU = logtheta_rnd(model.alpha, model.KU, model.numUsers);
logthetaM = logtheta_rnd(model.alpha, model.KM, model.numItems);

% Form struct containing sample
sample = struct('LambdaU', {LambdaU}, ...
                 'LambdaM', {LambdaM}, ...
                 'muU', {muU}, ...
                 'muM', {muM}, ...
                 'logthetaU', {logthetaU}, ...
                 'a', {a}, ...
                 'c', {c}, ...
                 'logthetaM', {logthetaM}, ...
                 'b', {b}, ...
                 'd', {d}, ...
                 'chi', {chi}, ...
                 'zU', {[]}, ...
                 'zM', {[]}, ...
                 'r', {[]});

if nargin > 1
    % Sample topics for all examples
    if model.KU > 0
        sample.zU = sampleVectorMex(exp(logthetaU), uIDs);
    end
    if model.KM > 0
        sample.zM = sampleVectorMex(exp(logthetaM), mIDs);
    end
    if (nargin > 3) && sampleRatings
        % Sample ratings for all uIDs/mIDs
        sample.r = normrnd(...
            m3f_tib_predictMex(uIDs, mIDs, sample, sample.zU,...
                                 sample.zM, [true, true, true]), model.sigmaSqd);
    end
end
end

% -----------------------------END OF CODE-------------------------------
