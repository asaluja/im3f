function new_mu_k = updateCRFMu(mu, n, resid, k, isplus, model)

if isplus
    new_mu_k = (mu(k) * (n(k) + model.sigmaSqd2sigmaSqd0) + resid) / (n(k) + 1 + model.sigmaSqd2sigmaSqd0);
else
    new_mu_k = (mu(k) * (n(k) + model.sigmaSqd2sigmaSqd0) - resid) / (n(k) - 1 + model.sigmaSqd2sigmaSqd0);
end