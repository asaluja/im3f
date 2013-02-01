/* 
  M3F_TIF_SAMPLEFACTORVECTORS Gibbs samples the A and B matrices from 
  posterior.
   
  Usage: 
     m3f_tif_sampleFactorVectors(data, model, samp, resids)

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - m3f_tif structure (see m3f_tif_initModel)
     samp - Current Gibbs sample of model variables
     resids - Differences between true ratings and rating predictions
     (see m3f_tif_gibbs)

  Outputs:
     This function modifies the "samp" structure input IN PLACE, 
     replacing the 'a' and 'b' samples.

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

// Sample factor vectors
// Function written from perspective of sampling user factor vectors
// Switch roles of user-item inputs to sample item factor vectors
void sampleFactorVectors(uint32_t* items, double* resids, const mxArray* exampsByUser, 
			 int numUsers, double invSigmaSqd, int numFacs, 
			 double* LambdaU, double* muU, double* a, double* b){
			 
   // Array of random number generators
   gsl_rng** rngs = getRngArray();  
 
   // Extract internals of jagged arrays
   uint32_t** userExamps;
   mwSize* userLens;
   unpackJagged(exampsByUser, &userExamps, &userLens, numUsers);
 
   int numFacsSqd = numFacs*numFacs;

   // BLAS constants
   char uplo[] = "U";
   char trans[] = "N";
   char diag[] = "N";
   int oneInt = 1;
   double oneDbl = 1;
   double zeroDbl = 0;

   // Compute muBase = LambdaU*muU
   double* muBase = mxMalloc(numFacs*sizeof(*muBase));
   dsymv(uplo, &numFacs, &oneDbl, LambdaU, &numFacs, muU, &oneInt, &zeroDbl, muBase, &oneInt);

   // Allocate memory for new mean and precision parameters
   double* muNew[MAX_NUM_THREADS];
   double* LambdaNew[MAX_NUM_THREADS];
   for(int thread = 0; thread < MAX_NUM_THREADS; thread++){
      muNew[thread] = mxMalloc(numFacs*sizeof(**muNew));
      LambdaNew[thread] = mxMalloc(numFacsSqd*sizeof(**LambdaNew));      
   }

#pragma omp parallel for
   for(int u = 0; u < numUsers; u++){
      int thread = omp_get_thread_num();
      // Initialize new mean to muBase
      dcopy(&numFacs, muBase, &oneInt, muNew[thread], &oneInt);
      // Initialize new precision to LambdaU
      dcopy(&numFacsSqd, LambdaU, &oneInt, LambdaNew[thread], &oneInt);
      
      // Iterate over user's examples
      mwSize len = userLens[u];
      uint32_t* examps = userExamps[u];
      for(int j = 0; j < len; j++){
	 uint32_t e = examps[j]-1;
	 int m = items[e]-1;

	 // Item vector for this rated item
	 double* bVec = b + m*numFacs;

	 // Compute posterior sufficient statistics for factor vector
	 // Add resid * bVec/sigmaSqd to muNew
	 double resid = resids[e];
	 resid *= invSigmaSqd;
	 daxpy(&numFacs, &resid, bVec, &oneInt, muNew[thread], &oneInt);

	 // Add bVec * bVec^t to LambdaNew
	 // Exploit symmetric structure of LambdaNew
	 dsyr(uplo, &numFacs, &invSigmaSqd, bVec, &oneInt, LambdaNew[thread], &numFacs);
      }
      
      // Compute upper Cholesky factor of LambdaNew
      int info;
      dpotrf(uplo, &numFacs, LambdaNew[thread], &numFacs, &info);

      // Solve for (LambdaNew)^-1*muNew using Cholesky factor
      dpotrs(uplo, &numFacs, &oneInt, LambdaNew[thread], &numFacs, muNew[thread], &numFacs, &info);

      // Sample vector of N(0,1) variables
      gsl_rng* rng = rngs[thread];
      double* aVec = a + u*numFacs;
      for(int f = 0; f < numFacs; f++)
	 aVec[f] = gsl_ran_gaussian(rng, 1);
      
      // Solve for (chol(LambdaNew,'U'))^-1*N(0,1)
      dtrtrs(uplo, trans, diag, &numFacs, &oneInt, LambdaNew[thread], &numFacs, aVec, &numFacs, &info);
      
      // Add muNew to aVec
      daxpy(&numFacs, &oneDbl, muNew[thread], &oneInt, aVec, &oneInt);
   }
   // Clean up
   mxFree(userExamps);
   mxFree(userLens);
   mxFree(muBase);
   for(int thread = 0; thread < MAX_NUM_THREADS; thread++){
      mxFree(muNew[thread]);
      mxFree(LambdaNew[thread]);
   }
}

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running m3f_tif_sampleFactorVectors\n");

   omp_set_num_threads(MAX_NUM_THREADS);

   // Extract input information
   const mxArray* data = prhs[0];
   uint32_t* users = (uint32_t*)mxGetData(mxGetField(data, 0, "users"));
   uint32_t* items = (uint32_t*)mxGetData(mxGetField(data, 0, "items"));
   const mxArray* exampsByUser = mxGetField(data, 0, "exampsByUser");
   const mxArray* exampsByItem = mxGetField(data, 0, "exampsByItem");
   const mxArray* model = prhs[1];
   int numUsers = (*mxGetPr(mxGetField(model, 0, "numUsers"))) + .5;
   int numItems = (*mxGetPr(mxGetField(model, 0, "numItems"))) + .5;
   double invSigmaSqd = 1/(*mxGetPr(mxGetField(model, 0, "sigmaSqd")));
   const mxArray* samp = prhs[2];
   const mxArray* LambdaUarray = mxGetField(samp, 0, "LambdaU");
   int numFacs = mxGetM(LambdaUarray);
   double* LambdaU = mxGetPr(LambdaUarray);
   double* muU = mxGetPr(mxGetField(samp, 0, "muU"));
   double* LambdaM = mxGetPr(mxGetField(samp, 0, "LambdaM"));
   double* muM = mxGetPr(mxGetField(samp, 0, "muM"));
   double* a = mxGetPr(mxGetField(samp, 0, "a"));
   double* b = mxGetPr(mxGetField(samp, 0, "b"));
   double* resids = mxGetPr(prhs[3]);

   if(numFacs > 0){
      // Sample user factor vectors   
      sampleFactorVectors(items, resids, exampsByUser,
			  numUsers, invSigmaSqd, numFacs, LambdaU,
			  muU, a, b);
      
      // Sample item factor vectors
      sampleFactorVectors(users, resids, exampsByItem,
			  numItems, invSigmaSqd, numFacs, LambdaM,
			  muM, b, a);
   }
}
