function [LambdaU, muU] = ...
      m3f_tif_sampleTopicHyperParams(a, W0inv, nu0, mu0, lambda0, numUserVecs, ...
                                  eyeSizeNumFacs)
%M3F_TIB_SAMPLETOPICHYPERPARAMS Gibbs sample hyperparameters of
%m3f_tif topic-indexed factor matrix. Written from perspective of
%sampling user hyperparameters.
%
% Usage:
%    [LambdaU, muU] = m3f_tif_sampleTopicHyperParams(a, W0inv, nu0, mu0,
%    lambda0, numUserVecs, eyeSizeNumFacs)
%
% Inputs:
%    a - Current topic-indexed factor matrix
%    W0inv - Inverse scale matrix of Normal-Wishart distribution
%    nu0 - Degrees of freedom of Normal-Wishart distribution
%    mu0 - Mean vector of Normal-Wishart distribution
%    lambda0 - Precision scaling factor of Normal-Wishart prior
%    numUserVecs - Num user topics times maximum user id
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
   % Sample user hyperparameters
   denom = (lambda0+numUserVecs);
   aBar = sum(sum(a,3),2)/numUserVecs;
   Winv = W0inv + lambda0*numUserVecs/denom *...
          (mu0-aBar)*(mu0-aBar)' - ...
          numUserVecs*aBar*aBar';
   for i = 1:size(a,3)
      Winv = Winv + a(:,:,i)*a(:,:,i)';
   end
   
   % Sample from wishart with identity scale and post-multiply by
   %   cholesky factor of scale
   LambdaU = wishrnd([], nu0 + numUserVecs, ...
                     solve_triu(cholproj(Winv), eyeSizeNumFacs));
   muU = randnorm(1,(lambda0*mu0+numUserVecs*aBar)/denom,...
                   [],(LambdaU*denom)\eyeSizeNumFacs);
else
   LambdaU = [];
   muU = [];
end

% -----------------------------END OF CODE-------------------------------
