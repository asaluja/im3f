/* 
  M3F_TIB_SAMPLEFACTORVECTORS Gibbs samples the A and B matrices from 
  posterior.
   
  Usage: 
     m3f_tib_sampleFactorVectors(data, model, samp, zU, zM, storeMeans)
     m3f_tib_sampleFactorVectors(data, model, samp, zU, zM, storeMeans, 
                                   [sampUserParams, sampItemParams])

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - m3f_tib structure (see m3f_tib_initModel)
     samp - Current Gibbs sample of model variables
     zU,zM - sampled user/item latent topics
     storeMeans - if true, posterior means are stored instead of samples
     sampleUserParams - OPTIONAL: if false, user params are not sampled
     sampleItemParams - OPTIONAL: if false, item params are not sampled

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
void sampleFactorVectors(uint32_t* items, double* vals, const mxArray* exampsByUser, 
			 ptrdiff_t KU, ptrdiff_t KM, ptrdiff_t numUsers, double invSigmaSqd, ptrdiff_t numFacs, 
			 double* LambdaU, double* muU, double* a, double* b, double* c, double* d, 
			 double chi, uint32_t* zU, uint32_t* zM, mxLogical storeMeans){
   // Array of random number generators
   gsl_rng** rngs = getRngArray();  

   // Extract internals of jagged arrays
   uint32_t** userExamps;
   mwSize* userLens;
   unpackJagged(exampsByUser, &userExamps, &userLens, numUsers);
 
   ptrdiff_t numFacsSqd = numFacs*numFacs;
 
   // BLAS constants
   char uplo[] = "U";
   char trans[] = "N";
   char diag[] = "N";
   ptrdiff_t oneInt = 1;
   double oneDbl = 1;
   double zeroDbl = 0;

   // Compute muBase = LambdaU*muU
   double* muBase = mxMalloc(numFacs*sizeof(*muBase));
   dsymv(uplo, &numFacs, &oneDbl, LambdaU, &numFacs, muU, &oneInt, &zeroDbl, muBase, &oneInt);

   // Allocate memory for new mean and precision parameters
   double* muNew[MAX_NUM_THREADS];
   double* LambdaNew[MAX_NUM_THREADS];
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++){
      muNew[thread] = mxMalloc(numFacs*sizeof(**muNew));
      LambdaNew[thread] = mxMalloc(numFacsSqd*sizeof(**LambdaNew));      
   }

#pragma omp parallel for
   for(ptrdiff_t u = 0; u < numUsers; u++){
      ptrdiff_t thread = omp_get_thread_num();
      // Initialize new mean to muBase
      dcopy(&numFacs, muBase, &oneInt, muNew[thread], &oneInt);
      // Initialize new precision to LambdaU
      dcopy(&numFacsSqd, LambdaU, &oneInt, LambdaNew[thread], &oneInt);
      
      // Iterate over user's examples
      mwSize len = userLens[u];
      uint32_t* examps = userExamps[u];
      for(ptrdiff_t j = 0; j < len; j++){
	 uint32_t e = examps[j]-1;
	 ptrdiff_t m = items[e]-1;

	 // Item vector for this rated item
	 double* bVec = b + m*numFacs;

	 // Compute posterior sufficient statistics for factor vector
	 // Add resid * bVec/sigmaSqd to muNew
	 double resid = vals[e] - chi;
     resid -= d[zU[e]-1];
     resid -= c[zM[e]-1];
	 resid *= invSigmaSqd;
	 daxpy(&numFacs, &resid, bVec, &oneInt, muNew[thread], &oneInt);

	 // Add bVec * bVec^t to LambdaNew
	 // Exploit symmetric structure of LambdaNew
	 dsyr(uplo, &numFacs, &invSigmaSqd, bVec, &oneInt, LambdaNew[thread], &numFacs);
      }
      
      // Compute upper Cholesky factor of LambdaNew
      ptrdiff_t info;
      dpotrf(uplo, &numFacs, LambdaNew[thread], &numFacs, &info);

      // Solve for (LambdaNew)^-1*muNew using Cholesky factor
      dpotrs(uplo, &numFacs, &oneInt, LambdaNew[thread], &numFacs, muNew[thread], &numFacs, &info);

      if(storeMeans){
	 // Use mean of distribution instead of sample
	 dcopy(&numFacs, muNew[thread], &oneInt, a + u*numFacs, &oneInt);
      }
      else{
	 // Sample vector of N(0,1) variables
	 gsl_rng* rng = rngs[thread];
	 double* aVec = a + u*numFacs;
	 for(ptrdiff_t f = 0; f < numFacs; f++)
	    aVec[f] = gsl_ran_gaussian(rng, 1);
	 
	 // Solve for (chol(LambdaNew,'U'))^-1*N(0,1)
	 dtrtrs(uplo, trans, diag, &numFacs, &oneInt, LambdaNew[thread], &numFacs, aVec, &numFacs, &info);

	 // Add muNew to aVec
	 daxpy(&numFacs, &oneDbl, muNew[thread], &oneInt, aVec, &oneInt);
      }
   }
   // Clean up
   mxFree(userExamps);
   mxFree(userLens);
   mxFree(muBase);
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++){
      mxFree(muNew[thread]);
      mxFree(LambdaNew[thread]);
   }
}

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running m3f_tib_sampleFactorVectors\n");

   omp_set_num_threads(MAX_NUM_THREADS);

   // Extract input information
   const mxArray* data = prhs[0];
   uint32_t* users = (uint32_t*)mxGetData(mxGetField(data, 0, "users"));
   uint32_t* items = (uint32_t*)mxGetData(mxGetField(data, 0, "items"));
   double* vals = mxGetPr(mxGetField(data, 0, "vals"));
   const mxArray* exampsByUser = mxGetField(data, 0, "exampsByUser");
   const mxArray* exampsByItem = mxGetField(data, 0, "exampsByItem");
   const mxArray* model = prhs[1];
   ptrdiff_t numUsers = (*mxGetPr(mxGetField(model, 0, "numUsers"))) + .5;
   ptrdiff_t numItems = (*mxGetPr(mxGetField(model, 0, "numItems"))) + .5;
   double invSigmaSqd = 1/(*mxGetPr(mxGetField(model, 0, "sigmaSqd")));
   const mxArray* samp = prhs[2];
   const mxArray* LambdaUarray = mxGetField(samp, 0, "LambdaU");
   ptrdiff_t numFacs = mxGetM(LambdaUarray);
   double* LambdaU = mxGetPr(LambdaUarray);
   double* muU = mxGetPr(mxGetField(samp, 0, "muU"));
   double* LambdaM = mxGetPr(mxGetField(samp, 0, "LambdaM"));
   double* muM = mxGetPr(mxGetField(samp, 0, "muM"));
   double* a = mxGetPr(mxGetField(samp, 0, "a"));
   double* b = mxGetPr(mxGetField(samp, 0, "b"));
   double* c = mxGetPr(mxGetField(samp, 0, "c"));
   double* d = mxGetPr(mxGetField(samp, 0, "d"));
   ptrdiff_t KU = mxGetN(mxGetField(samp, 0, "muD"));
   ptrdiff_t KM = mxGetN(mxGetField(samp, 0, "muC"));
   double chi = *(mxGetPr(mxGetField(samp, 0, "chi")));
   uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
   uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
   mxLogical storeMeans = *mxGetLogicals(prhs[5]);
   mxLogical* sampParams = NULL;
   if(nrhs > 6)
      sampParams = mxGetLogicals(prhs[6]);

   if(numFacs > 0){
      // Sample user factor vectors   
      if((sampParams == NULL) || sampParams[0]){
	 sampleFactorVectors(items, vals, exampsByUser, KU, KM,
			     numUsers, invSigmaSqd, numFacs, LambdaU,
			     muU, a, b, c, d, chi, zU, zM, storeMeans);
      }
      
      // Sample item factor vectors
      if((sampParams == NULL) || sampParams[1]){
	 sampleFactorVectors(users, vals, exampsByItem, KM, KU,
			     numItems, invSigmaSqd, numFacs, LambdaM,
			     muM, b, a, d, c, chi, zM, zU, storeMeans);  
      }
   }
}
