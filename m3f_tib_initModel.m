function [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM, gamma, beta, W0,...
                                       nu0, mu0, lambda0, alpha, sigmaSqd, ...
                                       sigmaSqd0, c0, d0, chi0)
%M3F_TIB_INITMODEL Initialize and return a TIB model structure.
%
% Usage:
%    [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM)
%    [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM, W0,
%              nu0, mu0, lambda0, alpha, sigmaSqd, sigmaSqd0, c0, d0, chi0)
%
% Inputs:
%    numUsers - Maximum user id
%    numItems - Maximum item id
%    numFacs - Number of static latent factors
%    KU - Number of user topics
%    KM - Number of item topics
%    See Mackey et al. for definitions of remaining free parameters:
%    W0 (numFacs x numFacs)
%    nu0 (1 x 1)
%    mu0 (numFacs x 1)
%    lambda0 (1 x 1)
%    alpha (1 x 1)
%    sigmaSqd (1 x 1)
%    sigmaSqd0 (1 x 1)
%    c0 (1 x 1)
%    d0 (1 x 1)
%    chi0 (1 x 1)
%    gammaU (1 x 1)
%    gammaM (1 x 1)
%
% Outputs:
%    model - TIB model structure initialized according to
%            given inputs and default values for unspecified free
%            parameters

% -----------------------------BEGIN CODE--------------------------------

model.numUsers = numUsers;
model.numItems = numItems;
model.KU = KU;
model.KM = KM;

% Default values for free parameters
model.W0 = eye(numFacs);
model.nu0 = numFacs;
model.mu0 = zeros(numFacs,1);
model.lambda0 = 10;
model.alpha = 10000;
model.sigmaSqd = .5;
model.sigmaSqd0 = .1;
model.invsigmaSqd = 1 / model.sigmaSqd;
model.invsigmaSqd0 = 1 / model.sigmaSqd0;
model.sigmaSqd2sigmaSqd0 = model.sigmaSqd / model.sigmaSqd0;
model.c0 = 0;
model.d0 = 0;
model.chi0 = 0;
model.gammaU = gamma;
model.gammaM = gamma;
model.betaU = beta;
model.betaM = beta;

% -----------------------------END OF CODE-------------------------------
