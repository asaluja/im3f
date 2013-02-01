function samp = m3f_tif_initSamp(initMode, model, data, testData)
%M3F_TIF_INITSAMP Return an initial "sample" from model initialized
%according to initMode.
%
% Usage:
%    [samp] = m3f_tif_initSamp(initMode, model, data, testData)
%
% Inputs:
%    initMode - Mode used to create initial sample
%               1 => Draw random sample from model prior
%               2 => Initialize hidden variables to model means
%               3 => Initialize static factors to MAP estimate trained
%               using stochastic gradient descent and set remaining
%               variables to model means
%    model - m3f_tif structure (see m3f_tif_initModel)
%    data - Dyadic data structure (see loadDyadicData)
%    testData - Dyadic test data structure (see loadDyadicData)
%
% Outputs:
%    samp - Populated sample structure with fields corresponding to
%           the hidden parameters and topics of the m3f_tif:
%           LambdaU (numFacs x numFacs)
%           LambdaM (numFacs x numFacs)
%           muU (numFacs x 1)
%           muM (numFacs x 1)
%           LambdaTildeU (numTopicFacs x numTopicFacs)
%           LambdaTildeM (numTopicFacs x numTopicFacs)
%           muTildeU (numTildeFacs x 1)
%           muTildeM (numTildeFacs x 1)
%           logthetaU (KU x numUsers)
%           a (numFacs x numUsers)
%           c (numTopicFacs x numUsers x KM)
%           xi (1 x numUsers)
%           logthetaM (KM x numItems)
%           b (numFacs x numItems)
%           d (numTopicFacs x numItems x KU)
%           chi (1 x numItems)
%           zU (numExamples x 1)
%           zM (numExamples x 1)

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

% Require at least one user and one item topic
assert(model.KU > 0);
assert(model.KM > 0);

%% Initialize sample according to initMode
if initMode < 2
   fprintf('Drawing first sample from model prior\n');

   % Sample initial estimates from model prior
   samp = m3f_tif_rnd(model, data.users, data.items, false);
   samp = rmfield(samp, 'r');
elseif initMode < 3
   fprintf('Initializing first sample with prior means\n');
   
   % Initialize to model means
   samp.LambdaU = model.nu0*model.W0;
   samp.LambdaM = model.nu0*model.W0;
   samp.muU = model.mu0; 
   samp.muM = model.mu0;     
   samp.LambdaTildeU = model.nuTilde0*model.WTilde0;
   samp.LambdaTildeM = model.nuTilde0*model.WTilde0;
   samp.muTildeU = model.muTilde0; 
   samp.muTildeM = model.muTilde0;     

   samp.a = repmat(samp.muU, 1, model.numUsers);
   samp.b = repmat(samp.muM, 1, model.numItems);
   
   samp.xi = repmat(model.xi0, 1, model.numUsers);
   samp.chi = repmat(model.chi0, 1, model.numItems);
   
   samp.c = repmat(samp.muTildeU, [1 model.numUsers model.KM]);
   samp.d = repmat(samp.muTildeM, [1 model.numItems model.KU]);
   
   samp.logthetaU = repmat(log(1.0/model.KU), model.KU, model.numUsers);
   samp.logthetaM = repmat(log(1.0/model.KM), model.KM, model.numItems);
   samp.zU = sampleVectorMex(exp(samp.logthetaU), data.users);
   samp.zM = sampleVectorMex(exp(samp.logthetaM), data.items);
   
elseif initMode < 4
   fprintf(['Initializing static factors to MAP estimates trained with ' ...
            'stochastic gradient descent\n']);
    
   %% Initialize to model means and then run PMF
   samp = m3f_tif_initSamp(2, model, data, testData);
   sgdFactorVectors(data, model, samp, 10, testData);      
else
   error('invalid init mode: %d', initMode);
end
% Order sample fields alphabetically
samp = orderfields(samp);

% -----------------------------END OF CODE-------------------------------
