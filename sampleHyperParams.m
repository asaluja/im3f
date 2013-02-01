function [LambdaU, muU] = ...
      sampleHyperParams(model, a, W0inv, numUsers, ...
                                  eyeSizeNumFacs)
%SAMPLEHYPERPARAMS Gibbs sample hyperparameters of m3f_tib
%factor matrix. Written from perspective of sampling user hyperparameters.
%
% Usage:
%    [LambdaU, muU] = sampleHyperParams(model, a, W0inv, numUsers,
%                                                 eyeSizeNumFacs)
%
% Inputs:
%    model - M3F model structure (see m3f_tib_initModel, m3f_tif_initModel)
%    a - Current factor matrix
%    W0inv - Inverse scale matrix of Wishart distribution
%    numUsers - Maximum user id
%    eyeSizeNumFacs - numFacs x numFacs identity matrix
%
% Outputs:
%    LambdaU - Sampled precision matrix for user factor vectors
%    muU - Sampled mean vector for user factor vectors

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

if size(eyeSizeNumFacs, 1) > 0
   %% Sample user hyperparameters
   denom = (model.lambda0+numUsers);
   aBar = mean(a,2);
   Winv = W0inv + a*a' - ...
          numUsers*aBar*aBar' + ...
          model.lambda0*numUsers/denom *...
          (model.mu0-aBar)*(model.mu0-aBar)';
   
   % Sample from Wishart with identity scale and post-multiply by
   % Cholesky factor of scale
   LambdaU = wishrnd([], model.nu0 + numUsers, ...
                     solve_triu(cholproj(Winv), eyeSizeNumFacs));
   muU = randnorm(1,(model.lambda0*model.mu0+numUsers*aBar)/denom,...
                   [],(LambdaU*denom)\eyeSizeNumFacs);
else
   LambdaU = [];
   muU = [];
end

% -----------------------------END OF CODE-------------------------------
