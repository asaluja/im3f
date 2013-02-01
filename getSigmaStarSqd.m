function sigmaStarSqd = getSigmaStarSqd(km, ku, nC, nD, model)

if km > length(nC)
    km
end
sigmaStarSqd = model.sigmaSqd + 1 / (model.invsigmaSqd0 + model.invsigmaSqd * nC(km)) + 1 / (model.invsigmaSqd0 + model.invsigmaSqd * nD(ku));