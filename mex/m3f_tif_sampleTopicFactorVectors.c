/* 
  M3F_TIF_SAMPLETOPICFACTORVECTORS Gibbs samples the C and D matrices from 
  posterior.
   
  Usage: 
     m3f_tif_sampleTopicFactorVectors(data, model, samp, zU, zM, resids)

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - m3f_tif structure (see m3f_tif_initModel)
     samp - Current Gibbs sample of model variables
     zU,zM - sampled user/item latent topics
     resids - Differences between true ratings and rating predictions
     (see m3f_tif_gibbs)

  Outputs:
     This function modifies the "samp" structure input IN PLACE, 
     replacing the 'c' and 'd' samples.

  ------------------------------------------------------------------------     

  Last revision: 12-July-2010

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
// Function written from perspective of sampling user factor vectors with cross-topics
// Switch roles of user-item inputs to sample item factor vectors
void sampleTopicFactorVectors(uint32_t* items, double* resids, const mxArray* exampsByUser,
			      int KU, int KM, int numUsers, int numItems, double invSigmaSqd, 
			      int numTopicFacs, double* LambdaU, double* muU, double* c, double* d, 
			      uint32_t* zU, uint32_t* zM){
   // Array of random number generators
   gsl_rng** rngs = getRngArray();  
 
   // Extract internals of jagged arrays
   uint32_t** userExamps;
   mwSize* userLens;
   unpackJagged(exampsByUser, &userExamps, &userLens, numUsers);

   int numTopicFacsSqd = numTopicFacs*numTopicFacs;
   int numTopicFacsTimesNumItems = numTopicFacs*numItems;
   int numTopicFacsTimesNumUsers = numTopicFacs*numUsers;

   // BLAS constants
   char uplo[] = "U";
   char trans[] = "N";
   char diag[] = "N";
   int oneInt = 1;
   double oneDbl = 1;
   double zeroDbl = 0;

   // Compute muBase = LambdaU*muU
   double* muBase = mxMalloc(numTopicFacs*sizeof(*muBase));
   dsymv(uplo, &numTopicFacs, &oneDbl, LambdaU, &numTopicFacs, muU, &oneInt, &zeroDbl, muBase, &oneInt);

   // Allocate memory for new mean and precision parameters
   double** muNew[MAX_NUM_THREADS];
   double** LambdaNew[MAX_NUM_THREADS];
   for(int thread = 0; thread < MAX_NUM_THREADS; thread++){
      muNew[thread] = mxMalloc(KM*sizeof(**muNew));
      LambdaNew[thread] = mxMalloc(KM*sizeof(**LambdaNew));
      for(int i = 0; i < KM; i++){
	 muNew[thread][i] = mxMalloc(numTopicFacs*sizeof(***muNew));
	 LambdaNew[thread][i] = mxMalloc(numTopicFacsSqd*sizeof(***LambdaNew));
      }
   }
// Commented out by Dongzhen to run the program
#pragma omp parallel for
   for(int u = 0; u < numUsers; u++){
      int thread = omp_get_thread_num();
      for(int i = 0; i < KM; i++){
	 // Initialize new mean to muBase
	 dcopy(&numTopicFacs, muBase, &oneInt, muNew[thread][i], &oneInt);
	 // Initialize new precision to LambdaU
	 dcopy(&numTopicFacsSqd, LambdaU, &oneInt, LambdaNew[thread][i], &oneInt);
      }

      // Iterate over user's examples
      mxArray* exampsArray = mxGetCell(exampsByUser, u);
      mwSize len = mxGetN(exampsArray);
      uint32_t* examps = (uint32_t*) mxGetData(exampsArray);
      for(int j = 0; j < len; j++){
	 uint32_t e = examps[j]-1;
	 int m = items[e]-1;
	 int userTop = zU[e]-1;
	 int itemTop = zM[e]-1;

	 // Item vector for this rated item
	 double* dVec = d + m*numTopicFacs + userTop*numTopicFacsTimesNumItems;

	 // Compute posterior sufficient statistics for factor vector
	 // Add resid * dVec/sigmaSqd to muNew
	 double resid = resids[e];
	 resid *= invSigmaSqd;
	 daxpy(&numTopicFacs, &resid, dVec, &oneInt, muNew[thread][itemTop], &oneInt);

	 // Add (dVec * dVec^t)/sigmaSqd to LambdaNew
	 // Exploit symmetric structure of LambdaNew
	 dsyr(uplo, &numTopicFacs, &invSigmaSqd, dVec, &oneInt, LambdaNew[thread][itemTop], 
	      &numTopicFacs);
      }
      
      for(int i = 0; i < KM; i++){
	 // Compute upper Cholesky factor of LambdaNew
	 int info;
	 dpotrf(uplo, &numTopicFacs, LambdaNew[thread][i], &numTopicFacs, &info);
	 
	 // Solve for (LambdaNew)^-1*muNew using Cholesky factor
	 dpotrs(uplo, &numTopicFacs, &oneInt, LambdaNew[thread][i], &numTopicFacs, muNew[thread][i], 
		&numTopicFacs, &info);
	 
	 // Sample vector of N(0,1) variables
	 gsl_rng* rng = rngs[thread];
	 double* cVec = c + u*numTopicFacs + i*numTopicFacsTimesNumUsers;
	 for(int f = 0; f < numTopicFacs; f++)
	    cVec[f] = gsl_ran_gaussian(rng, 1);
	 
	 // Solve for (chol(LambdaNew,'U'))^-1*N(0,1)
	 dtrtrs(uplo, trans, diag, &numTopicFacs, &oneInt, LambdaNew[thread][i], 
		&numTopicFacs, cVec, &numTopicFacs, &info);
	 
	 // Add muNew to aVec
	 daxpy(&numTopicFacs, &oneDbl, muNew[thread][i], &oneInt, cVec, &oneInt);
      }
   }
   // Clean up
   mxFree(userExamps);
   mxFree(userLens);
   mxFree(muBase);
   for(int thread = 0; thread < MAX_NUM_THREADS; thread++){
      for(int i = 0; i < KM; i++){
	 mxFree(muNew[thread][i]);
	 mxFree(LambdaNew[thread][i]);
      }
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
   int KU = (*mxGetPr(mxGetField(model, 0, "KU"))) + .5;
   int KM = (*mxGetPr(mxGetField(model, 0, "KM"))) + .5;
   int numUsers = (*mxGetPr(mxGetField(model, 0, "numUsers"))) + .5;
   int numItems = (*mxGetPr(mxGetField(model, 0, "numItems"))) + .5;
   double invSigmaSqd = 1/(*mxGetPr(mxGetField(model, 0, "sigmaSqd")));
   const mxArray* samp = prhs[2];
   // Note: Tilde parameters are loaded
   const mxArray* LambdaUarray = mxGetField(samp, 0, "LambdaTildeU");
   int numTopicFacs = mxGetM(LambdaUarray);
   double* LambdaU = mxGetPr(LambdaUarray);
   double* muU = mxGetPr(mxGetField(samp, 0, "muTildeU"));
   double* LambdaM = mxGetPr(mxGetField(samp, 0, "LambdaTildeM"));
   double* muM = mxGetPr(mxGetField(samp, 0, "muTildeM"));
   double* c = mxGetPr(mxGetField(samp, 0, "c"));
   double* d = mxGetPr(mxGetField(samp, 0, "d"));
   uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
   uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
   double* resids = mxGetPr(prhs[5]);

   if(numTopicFacs > 0){
      if(KU > 0 && KM > 0){
	 // Sample user factor vectors   
	 sampleTopicFactorVectors(items, resids, exampsByUser, KU, KM, 
			     numUsers, numItems, invSigmaSqd, numTopicFacs, 
			     LambdaU, muU, c, d,  zU, 
			     zM);
	 sampleTopicFactorVectors(users, resids, exampsByItem, KM, KU,
			     numItems, numUsers, invSigmaSqd, numTopicFacs, 
			     LambdaM, muM, d, c, zM, 
			     zU);
      }
   }
}
