/*
 * M3F_TIB_PREDICTTONEYMEX Generates predictions given model samples.
 *
 * Usage:
 * [preds] = m3f_tib_predictMex(users, items, samples, zU, zM, [addBase, addCoffsets, addDoffsets], topicModel)
 *
 * Inputs:
 * users, items - numerical arrays for each (user,item) pair to be predicted
 * samples - data from gibbs sampling
 * zU, zM - sampled user and item topics.
 * if empty, then topics will be integrated out
 * addBase - 1 or 0: do or do not add matrix factorization contribution
 * <a,b> and global offset chi to prediction
 * addCoffsets - 1 or 0: do or do not add offset param c to prediction
 * addDoffsets - 1 or 0: do or do not add offset param d to prediction
 * topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}
 *
 * Outputs:
 * preds - numerical predictions for given dyads
 *
 * Notes:
 * This function is used not just for computing posterior mean predictions,
 * but also for computing partial residuals during gibbs sampling.
 * Also, there is no checking for invalid inputs...
 *
 * References:
 * Mackey, Weiss, and Jordan, "Mixed Membership Matrix Factorization,"
 * International Conference on Machine Learning, 2010.
 *
 * ------------------------------------------------------------------------
 *
 * Last revision: 2-July-2010
 *
 * Authors: Lester Mackey and David Weiss
 * License: MIT License
 *
 * Copyright (c) 2010 Lester Mackey & David Weiss
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ------------------------------------------------------------------------
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include "mexCommon.h"
#include "mexUtil.h"

double getIntegratedOffset(uint32_t u, uint32_t j, ptrdiff_t KU, double* nD, double* muD) {
    return 0.0;
}
// Add offset predictions to preds
// Function written from perspective of predicting offsets based on user topics
// Switch roles of user-item inputs to predict based on item topics
void addOffsets(int topicModel, uint32_t* users, uint32_t* items, ptrdiff_t KU, ptrdiff_t numItems, ptrdiff_t numExamples,
        double* nD, double* muD, uint32_t* zU, double* preds){
    // Add in d offset contribution to preds
    if(zU != NULL){
        // Use given topics
#pragma omp parallel for
        for(mwSize e = 0; e < numExamples; e++) {
            uint32_t item = 0;
            switch (topicModel) {
                case 1:
                    item = 0;
                    //mexPrintf("Using item %d, muD=%f\n",item,muD[item*KU+(zU[e]-1)]);
                    preds[e] += muD[item*KU + (zU[e]-1)];
                    break;
                case 2:
                    //if (topicModel == 2)
                    item = items[e]-1;
                    //mexPrintf("Using item %d, muD=%f\n",item,muD[item*KU+(zU[e]-1)]);
                    preds[e] += muD[item*KU + (zU[e]-1)];
                    break;
                case 3: // TODO
                    break;
            }
        }
    }
    else if(KU > 1){
        // Integrate out topics
        // Calculate expected bias
        mxArray* array = mxCreateDoubleMatrix(numItems, 1, mxREAL);
        double* meanD = mxGetPr(array);
#pragma omp parallel for   
        for(mwSize e = 0; e < numItems; e++){
            double* muDPtr = muD + e*KU;
            double* nDPtr = nD + e*KU;
            double total_nD = 0;
            for (ptrdiff_t i = 0; i < KU; i++) 
                total_nD += nDPtr[i];
            //mexPrintf("total_nD: %f\n",total_nD);
            
            if (total_nD == 0)
                meanD[e] = 0;
            else 
                for (ptrdiff_t i = 0; i < KU; i++) 
                    meanD[e] += muDPtr[i]*nDPtr[i] / total_nD;                            
        }

#pragma omp parallel for         
        for(mwSize e = 0; e < numExamples; e++){
            preds[e] += meanD[items[e]-1];
        }
    }
    else{
        // Only one topic exists
#pragma omp parallel for
        for(mwSize e = 0; e < numExamples; e++){
            preds[e] += muD[(items[e]-1)*KU];
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    mexPrintf("Running m3f_tib_predictToneyMex\n");
    
    omp_set_num_threads(MAX_NUM_THREADS);
    
    // Extract input information
    uint32_t* users = (uint32_t*)mxGetData(prhs[0]);
    mwSize numExamples = mxGetM(prhs[0]);
    uint32_t* items = (uint32_t*)mxGetData(prhs[1]);
    const mxArray* samples = prhs[2];
    // Extract number of samples
    mwSize numSamples = mxGetM(samples);
    mwSize dimN = mxGetN(samples);
    if(dimN > numSamples)
        numSamples = dimN;
    double chi = *mxGetPr(mxGetField(samples, 0, "chi"));
    ptrdiff_t KU = mxGetM(mxGetField(samples, 0, "muD"));
    ptrdiff_t KM = mxGetM(mxGetField(samples, 0, "muC"));
    ptrdiff_t numItems = mxGetN(mxGetField(samples, 0, "muD"));
    ptrdiff_t numUsers = mxGetN(mxGetField(samples, 0, "muC"));    
    ptrdiff_t numFacs = mxGetM(mxGetField(samples, 0, "a"));
    uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
    uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
    mxLogical* add = mxGetLogicals(prhs[5]);
    mxLogical addBase = add[0];
    mxLogical addC = add[1];
    mxLogical addD = add[2];
    double* vals = mxGetPr(prhs[6]);
    int topicModel = (int)(vals[0]);
    
    mexPrintf("Predicting with topicModel = %d\n",topicModel); 
    mexPrintf("KU=%d,KM=%d,numItems=%d,numUsers=%d\n",KU,KM,numItems,numUsers);
    
    // Prepare output
    // New array auto-initialized to zeros
    plhs[0] = mxCreateDoubleMatrix(numExamples, 1, mxREAL);
    double* preds = mxGetPr(plhs[0]);
    
    ptrdiff_t blasStride = 1;
    
    // Form predictions under each sample
    for(ptrdiff_t t = 0; t < numSamples; t++){
        double* muC = mxGetPr(mxGetField(samples, t, "muC"));
        double* muD = mxGetPr(mxGetField(samples, t, "muD"));
        double* nC = mxGetPr(mxGetField(samples, t, "nC"));
        double* nD = mxGetPr(mxGetField(samples, t, "nD"));
        double* a = mxGetPr(mxGetField(samples, t, "a"));
        double* b = mxGetPr(mxGetField(samples, t, "b"));
        double* c = mxGetPr(mxGetField(samples, t, "c"));
        double* d = mxGetPr(mxGetField(samples, t, "d"));
        // Incorporate d offsets into prediction
        if(KU > 0 && addD) 
            addOffsets(topicModel, users, items, KU, numItems, numExamples, nD, muD, zU, preds);
        
        // Incorporate c offsets into prediction
        if(KM > 0 && addC) {
            addOffsets(topicModel, items, users, KM, numUsers, numExamples, nC, muC, zM, preds);
        }
        // Incorporate MF prediction and global offset into prediction
        if(addBase){
            if(numFacs > 0){
#pragma omp parallel for
                for(mwSize e = 0; e < numExamples; e++){
                    double* aVec = a + numFacs*(users[e]-1);
                    double* bVec = b + numFacs*(items[e]-1);

                    preds[e] += chi + ddot(&numFacs, aVec, &blasStride, bVec, &blasStride);
                }
            }
            else{
                for(mwSize e = 0; e < numExamples; e++)
                    preds[e] += chi;
            }
        }
    }
    if(numSamples > 1){
        // Average over all sample predictions
        for(mwSize e = 0; e < numExamples; e++)
            preds[e] /= numSamples;
    }
    mexPrintf("Finished m3f_tib_predictMex\n");
}
