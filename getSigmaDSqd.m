function sigmaDSqd = getSigmaDSqd(ku, nD, model)

sigmaDSqd = 1 / (model.invsigmaSqd0 + model.invsigmaSqd * nD(ku));