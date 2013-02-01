function [samp] = m3f_tib_initSampCRF(model, data, testData)
%M3F_TIB_INITSAMP Return an initial "sample" from model initialized
%according to initMode.
%
% Usage:
%    [samp] = m3f_tib_initSamp(initMode, model, data, testData)
%
% Inputs:
%    topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}
%    initMode - Mode used to create initial sample
%               1 => Draw random sample from model prior
%               2 => Initialize hidden variables to model means
%               3 => Initialize static factors to MAP estimate trained
%               using stochastic gradient descent and set remaining
%               variables to model means
%    model - m3f_tib structure (see m3f_tib_initModel)
%    data - Dyadic data structure (see loadDyadicData)
%    testData - Dyadic test data structure (see loadDyadicData)
%
% Outputs:
%    samp - Populated sample structure with fields corresponding to
%           the hidden parameters and topics of the m3f_tib:
%           LambdaU (numFacs x numFacs)
%           LambdaM (numFacs x numFacs)
%           muU (numFacs x 1)
%           muM (numFacs x 1)
%           logthetaU (KU x numUsers)
%           a (numFacs x numUsers)
%           c (KM x numUsers)
%           logthetaM (KM x numItems)
%           b (numFacs x numItems)
%           d (KU x numItems)
%           chi (1 x 1)
%           zU (numExamples x 1)
%           zM (numExamples x 1)
%           muC (KM x numUsers or 1)
%           muD (KU x numItems or 1)
%           nC (KM x numUsers or 1)
%           nD (KU x numItems or 1)
%           

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

%% Initialize sample according to initMode
if initMode < 2
   fprintf('Drawing first sample from model prior\n');
   % Sample initial estimates from model prior
   samp = m3f_tib_rnd(model, data.users, data.items, false);
   samp = rmfield(samp, 'r');
elseif initMode < 3
   fprintf('Initializing first sample with prior means\n');
   
   % Initialize parameters to model means
   samp.LambdaU = model.nu0*model.W0;
   samp.LambdaM = model.nu0*model.W0;
   samp.muU = model.mu0; 
   samp.muM = model.mu0; 
   samp.a = repmat(samp.muU, 1, model.numUsers);
   samp.b = repmat(samp.muM, 1, model.numItems);
   samp.c = repmat(model.c0, model.KM, model.numUsers);
   samp.d = repmat(model.d0, model.KU, model.numItems);
   samp.chi = model.chi0;
   
   % Sample topics for all examples
   if model.KU > 0 
      samp.logthetaU = repmat(log(1.0/model.KU), model.KU, ...
                              model.numUsers);
      samp.zU = sampleVectorMex(exp(samp.logthetaU), data.users);
   else
      samp.logthetaU = [];
      samp.zU = [];
   end
   if model.KM > 0
      samp.logthetaM = repmat(log(1.0/model.KM), model.KM, ...
                              model.numItems);
      samp.zM = sampleVectorMex(exp(samp.logthetaM), data.items);
   else
      samp.logthetaM = [];
      samp.zM = [];
   end
   
   % Toney 
   % Initialize muC, muD, l, crpM, crpU
   
   switch topicModel
       case 'shcrp'
           samp.muC = repmat(model.c0, model.KM, 1);
           samp.muD = repmat(model.c0, model.KU, 1);
           samp.nC = repmat(0, model.KM, 1);
           samp.nD = repmat(0, model.KU, 1);
       case 'secrp'
           samp.muC = repmat(model.c0, model.KM, model.numUsers);
           samp.muD = repmat(model.c0, model.KU, model.numItems);
           samp.nC = repmat(0, model.KM, model.numUsers);
           samp.nD = repmat(0, model.KU, model.numItems);
       case 'crf'
           error('Not implemented yet');
   end
   
   
   
   % TODO sample Z's from scratch, remove above z sampling phase
   for ee = 1:length(data.vals)
       jj = data.items(ee); % item id
       uu = data.users(ee); % user id
       ii = samp.zM(ee); % item topic
       ii_u = samp.zU(ee); % user topic
       
       resid = data.vals(ee) - model.chi0 - samp.a(:,uu)' * samp.b(:,jj);
       
       % Update muC, muD       
       switch topicModel
           case {'shcrp','secrp'} 
               if strcmp(topicModel,'shcrp')==1 % Only update vector muC, muD, so uu=1, jj=1
                   uu = 1; jj = 1;
               end
               residC = resid - samp.muD(ii_u,jj);
               residD = resid - samp.muC(ii,uu);
               [new_muC, new_nC] = updateMuC(samp.muC, samp.nC, uu, ii, residC, model.invsigmaSqd0, model.invsigmaSqd, true);
               samp.muC(ii,uu) = new_muC; samp.nC(ii,uu) = new_nC;
               [new_muD, new_nD] = updateMuC(samp.muD, samp.nD, jj, ii_u, residD, model.invsigmaSqd0, model.invsigmaSqd, true);
               samp.muD(ii_u,jj) = new_muD; samp.nD(ii_u,jj) = new_nD;
           case 'crf'
               error('Not implemented yet');
       end
   end
   
   
   
   
elseif initMode < 4
   fprintf(['Initializing static factors to MAP estimates trained with ' ...
            'stochastic gradient descent\n']);

   %% Initialize to model means and then learn MAP estimates of static factors
   samp = m3f_tib_initSamp(topicModel, 2, model, data, testData);
   sgdFactorVectors(data, model, samp, 10, testData);
   
else
   error('invalid init mode: %d', initMode);
end
% Order sample fields alphabetically
samp = orderfields(samp);