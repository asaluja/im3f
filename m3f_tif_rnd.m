function [sample] = m3f_tif_rnd(model, uIDs, mIDs, sampleRatings)
%M3F_TIF_RND Sample from m3f_tif prior.
%
% Usage:
%    [sample] = m3f_tif_rnd(model)
%    [sample] = m3f_tif_rnd(model, uIDs, mIDs)
%    [sample] = m3f_tif_rnd(model, uIDs, mIDs, sampleRatings)
%
% Inputs:
%    model - m3f_tif structure (see m3f_tif_initModel)
%    uIDs - Array of user ids. If present, user and item topics will be
%    sampled for each user id, item id pair.
%    mIDs - Array of item ids. If present, user and item topics will be
%    sampled for each user id, item id pair.
%    sampleRatings - Logical. If true, ratings will be sampled for each
%    user id, item id pair.
%
% Outputs:
%    sample - Populated sample structure, drawn from model prior (see
%    m3f_tif_initSamp)

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
function theta = theta_rnd(alpha, KU, numUsers)
if KU > 1
   theta = dirichlet_sample(repmat(alpha/KU, KU, 1), numUsers);
elseif KU > 0
   theta = ones(1, numUsers);
else
   theta = [];
end
end

% Function for sampling factors and factor hyperparameterS
function [LambdaU, LambdaM, muU, muM, a, b] = ...
    factor_rnd(W0, nu0, mu0, lambda0, numUsers, numItems)
eyeNumFacs = eye(size(W0));
if nu0 > 0
   LambdaU = wishrnd(W0, nu0);
   LambdaM = wishrnd(W0, nu0);
else
   LambdaU = W0*nu0;
   LambdaM = W0*nu0;
end
LambdaUinv = LambdaU\eyeNumFacs;
LambdaMinv = LambdaM\eyeNumFacs;
muU = randnorm(1,mu0,[], LambdaUinv/lambda0);
muM = randnorm(1,mu0,[], LambdaMinv/lambda0);
a = randnorm(numUsers,muU,[],LambdaUinv);
b = randnorm(numItems,muM,[],LambdaMinv);
end

%% Extract model information
numUsers = model.numUsers;
numItems = model.numItems;
% Find number of item/user vector topics
% User topics index item factors
numUserVecTopics = model.KM;
numItemVecTopics = model.KU;

%% Sample all model variables
% Sample static factors and factor hyperparameters
[LambdaU, LambdaM, muU, muM, a, b] = ...
    factor_rnd(model.W0, model.nu0, model.mu0, model.lambda0,...
               numUsers, numItems);

% Sample biases
xi = normrnd(model.xi0, model.sigmaSqd0, 1, numUsers);
chi = normrnd(model.chi0, model.sigmaSqd0, 1, numItems);

% Sample topic-indexed factors and factor hyperparameters
[LambdaTildeU, LambdaTildeM, muTildeU, muTildeM, c, d] = ...
    factor_rnd(model.WTilde0, model.nuTilde0, model.muTilde0, ...
               model.lambdaTilde0, model.KU, model.KM);
eyeNumTopicFacs = eye(size(model.WTilde0));
LambdaTildeUinv = LambdaTildeU\eyeNumTopicFacs;
LambdaTildeMinv = LambdaTildeM\eyeNumTopicFacs;
c = zeros(size(LambdaTildeU,1), numUsers, model.KM);
for i = 1:model.KM
    c(:,:,i) = randnorm(numUsers,muTildeU,[],LambdaTildeUinv);
end
d = zeros(size(LambdaTildeM,1), numItems, model.KU);
for i = 1:model.KU
    d(:,:,i) = randnorm(numItems,muTildeM,[],LambdaTildeMinv);
end

% Sample topic parameters
thetaU = theta_rnd(model.alpha, model.KU, numUsers);
thetaM = theta_rnd(model.alpha, model.KM, numItems);

% Form struct of containing sample
sample = struct('LambdaU', {LambdaU}, ...
                 'LambdaM', {LambdaM}, ...
                 'muU', {muU}, ...
                 'muM', {muM}, ...
                 'LambdaTildeU', {LambdaTildeU}, ...
                 'LambdaTildeM', {LambdaTildeM}, ...
                 'muTildeU', {muTildeU}, ...
                 'muTildeM', {muTildeM}, ...
                 'logthetaU', {log(thetaU)}, ...
                 'a', {a}, ...
                 'xi', {xi}, ...
                 'c', {c}, ...
                 'logthetaM', {log(thetaM)}, ...
                 'b', {b}, ...
                 'chi', {chi}, ...
                 'd', {d}, ...
                 'zU', {[]}, ...
                 'zM', {[]}, ...
                 'r', {[]});

if nargin > 1
   % Sample topics for all examples
   if model.KU > 0
      sample.zU = sampleVectorMex(thetaU, uIDs);
   end
   if model.KM > 0
      sample.zM = sampleVectorMex(thetaM, mIDs);
   end
   if (nargin > 3) && sampleRatings
      % Sample ratings for all uIDs/mIDs
      sample.r = normrnd(...
          m3f_tif_predictMex(uIDs, mIDs, sample, sample.zU,...
                               sample.zM, [true, true, true]),...
          model.sigmaSqd);
   end
end
end

% -----------------------------END OF CODE-------------------------------
