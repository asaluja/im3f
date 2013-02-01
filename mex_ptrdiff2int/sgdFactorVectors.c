/* 
  SGDFACTORVECTORS Form MAP estimates of static factor vectors using 
  stochastic gradient descent
   
  Usage: 
     sgdFactorVectors(data, model, samp, numRounds)
     sgdFactorVectors(data, model, samp, numRounds, testData)

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - M3F model structure (see *_initModel)
     samp - Current Gibbs sample of model variables
     numRounds - Number of rounds to run
     testData - OPTIONAL: Test dataset structure

  Outputs:
     This function modifies the "samp" structure input IN PLACE, 
     replacing the 'a' and 'b' values with those estimated using SGD.

  ------------------------------------------------------------------------     

  Last revision: 2-July-2010

  Authors: Lester Mackey and David Weiss
  License: MIT License

  Copyright (c) 2010 Lester Mackey & David Weiss

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:
  
  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  ------------------------------------------------------------------------
*/

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include "mexCommon.h"

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running sgdFactorVectors\n");

   omp_set_num_threads(MAX_NUM_THREADS);

   // Array of random number generators
   gsl_rng** rngs = getRngArray();  

   // Extract input information
   const mxArray* data = prhs[0];
   uint32_t* users = (uint32_t*)mxGetData(mxGetField(data, 0, "users"));
   uint32_t* items = (uint32_t*)mxGetData(mxGetField(data, 0, "items"));
   const mxArray* valsArray = mxGetField(data, 0, "vals");
   double* vals = mxGetPr(valsArray);
   ptrdiff_t numExamples = mxGetM(valsArray);
   const mxArray* model = prhs[1];
   ptrdiff_t numUsers = (*mxGetPr(mxGetField(model, 0, "numUsers"))) + .5;
   ptrdiff_t numItems = (*mxGetPr(mxGetField(model, 0, "numItems"))) + .5;
   const mxArray* samp = prhs[2];
   ptrdiff_t numFacs = mxGetM(mxGetField(samp, 0, "LambdaU"));
   double* a = mxGetPr(mxGetField(samp, 0, "a"));
   double* b = mxGetPr(mxGetField(samp, 0, "b"));
   double chi = *(mxGetPr(mxGetField(samp, 0, "chi"))); 
   ptrdiff_t numRounds = *(mxGetPr(prhs[3])) + .5;
   uint32_t *testUsers, *testItems;
   double *testVals;
   ptrdiff_t numTestExamples = 0;
   if(nrhs > 4) {
      const mxArray* testData = prhs[4];
      testUsers = (uint32_t*)mxGetData(mxGetField(testData, 0, "users"));
      testItems = (uint32_t*)mxGetData(mxGetField(testData, 0, "items"));
      const mxArray* testValsArray = mxGetField(testData, 0, "vals");
      testVals = mxGetPr(testValsArray);
      numTestExamples = mxGetM(testValsArray);
   }

   // Learning rate
   double lRate = .007;
   // Ridge coefficients 
   double ridgeU = .002;
   double ridgeM = .002;
   // Shrinkage factors for vector updates
   double shrinkU = 1-lRate*ridgeU;
   double shrinkM = 1-lRate*ridgeM;

   // Sample initial factor vectors from spherical Gaussian
   double variance = .001;
#pragma omp parallel for
   for(ptrdiff_t u = 0; u < numUsers; u++){
      gsl_rng* rng = rngs[omp_get_thread_num()];
      double* aVec = a + u*numFacs;
      for(ptrdiff_t f = 0; f < numFacs; f++)
	 aVec[f] = gsl_ran_gaussian(rng, variance);
   }
#pragma omp parallel for
   for(ptrdiff_t m = 0; m < numItems; m++){
      gsl_rng* rng = rngs[omp_get_thread_num()];
      double* bVec = b + m*numFacs;
      for(ptrdiff_t f = 0; f < numFacs; f++)
	 bVec[f] = gsl_ran_gaussian(rng, variance);
   }

   mexPrintf("SGD Factor Vectors numFacs=%d\n", numFacs);

   // Optimize factor vectors via gradient descent
   ptrdiff_t blasStride = 1;
   double* aVecCopy = mxMalloc(numFacs*sizeof(*aVecCopy)); 
   for(ptrdiff_t t = 1; t <= numRounds; t++){
      // Perform one pass through dataset
      for(ptrdiff_t e = 0; e < numExamples; e++){
	 ptrdiff_t u = users[e]-1;
	 ptrdiff_t m = items[e]-1;
	 
	 // Factor vectors for this rating
	 double* aVec = a + u*numFacs;
	 double* bVec = b + m*numFacs;
	 dcopy(&numFacs, aVec, &blasStride, aVecCopy, &blasStride);

	 // Compute prediction residual      
	 double resid = vals[e] - ddot(&numFacs, aVec, &blasStride, bVec, &blasStride) - chi;
	 
	 // Multiply residual by learning rate
	 resid *= lRate;
	 
	 // Perform gradient updates
	 dscal(&numFacs, &shrinkU, aVec, &blasStride);
	 daxpy(&numFacs, &resid, bVec, &blasStride, aVec, &blasStride);
	 dscal(&numFacs, &shrinkM, bVec, &blasStride);
	 daxpy(&numFacs, &resid, aVecCopy, &blasStride, bVec, &blasStride);
      }

      if(numTestExamples > 0) {
	 // Report error on test data
	 double rmse = 0;
	 double mae = 0;
#pragma omp parallel for reduction(+:rmse,mae)
	 for(ptrdiff_t e = 0; e < numTestExamples; e++){
	    ptrdiff_t u = testUsers[e]-1;
	    ptrdiff_t m = testItems[e]-1;
	    
	    // Factor vectors for this rating
	    double* aVec = a + u*numFacs;
	    double* bVec = b + m*numFacs;
	    
	    // Compute prediction residual      
	    double resid = testVals[e] - ddot(&numFacs, aVec, &blasStride, bVec, &blasStride) - chi;
	    rmse += resid*resid;
	    mae += ((resid>0)?resid:-resid);
	 }
	 mexPrintf("Round %d Eval:\n\tTest RMSE = %g, Test MAE = %g\n", 
		   t, sqrt(rmse/numTestExamples), mae/numTestExamples);
      }
   }
   mxFree(aVecCopy);
}
