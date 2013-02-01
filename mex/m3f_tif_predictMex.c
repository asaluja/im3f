/* 
  M3F_TIF_PREDICTMEX Generates predictions given model samples.
  
  Usage: 
     [preds] = m3f_tif_predictMex(users, items, samples, zU, zM,
                                    [addBase addTopicFactors addBiases])
 
  Inputs: 
     users, items - numerical arrays for each (user,item) pair to be predicted 
     samples - data from gibbs sampling 
     zU, zM - sampled user and item topics. 
              if empty, then topics will be integrated out 
     addBase - 1 or 0: do or do not add static factor contribution
     <a,b> and global offset chi to prediction 
     addTopicFactors - 1 or 0: do or do not add topic factor contribution 
     <c,d> to prediction 
     addBiases - 1 or 0: do or do not add biases chi and xi to prediction
 
  Outputs: 
     preds - numerical predictions for given dyads
  
  Notes: 
     This function is used not just for computing posterior mean predictions, 
     but also for computing partial residuals during gibbs sampling. 
     Also, there is no checking for invalid inputs...

  References:
     Mackey, Weiss, and Jordan, "Mixed Membership Matrix Factorization,"
     International Conference on Machine Learning, 2010.

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
#include <omp.h>
#include "mex.h"
#include "blas.h"
#include "mexCommon.h"
#include "mexUtil.h"

// Compute the expected topic factor prediction by integrating over topics
double integrateFactorVectors(double *cVec, double *dVec, 
			      double *logThetaPtrU, double *logThetaPtrM, 
			      int numTopicFacs, int numTopicFacsTimesNumUsers,
			      int numTopicFacsTimesNumItems, int KU, int KM) {

   int blasStride = 1;
   double y = 0;   
   for (int ku = 0; ku < KU; ku++) {
      for (int km = 0; km < KM; km++) {	 
	 double pu = exp(logThetaPtrU[ku]);
	 double pm = exp(logThetaPtrM[km]);	 
	 y += pu*pm*ddot(&numTopicFacs, cVec + numTopicFacsTimesNumUsers*km, 
			 &blasStride, dVec + numTopicFacsTimesNumItems*ku, 
			 &blasStride);
      }
   }
   
   return y;
}

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
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
   int KU = mxGetM(mxGetField(samples, 0, "logthetaU"));
   int KM = mxGetM(mxGetField(samples, 0, "logthetaM"));
   int numFacs = mxGetM(mxGetField(samples, 0, "a"));
   int numUsers = mxGetN(mxGetField(samples, 0, "a"));
   int numItems = mxGetN(mxGetField(samples, 0, "b"));
   int numTopicFacs = mxGetM(mxGetField(samples, 0, "LambdaTildeU"));
   uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
   uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
   mxLogical* add = mxGetLogicals(prhs[5]);
   mxLogical addBase = add[0];
   mxLogical addTopicFactors = add[1];
   mxLogical addBiases = add[2];

   // Prepare output
   // New array auto-initialized to zeros
   plhs[0] = mxCreateDoubleMatrix(numExamples, 1, mxREAL);
   double* preds = mxGetPr(plhs[0]);

   int blasStride = 1;
  
   for(int t = 0; t < numSamples; t++){
      double* logthetaU = mxGetPr(mxGetField(samples, t, "logthetaU"));
      double* logthetaM = mxGetPr(mxGetField(samples, t, "logthetaM"));
      double* a = mxGetPr(mxGetField(samples, t, "a"));
      double* b = mxGetPr(mxGetField(samples, t, "b"));
      double* c = mxGetPr(mxGetField(samples, t, "c"));
      double* d = mxGetPr(mxGetField(samples, t, "d"));
      double* xi = mxGetPr(mxGetField(samples, t, "xi"));
      double* chi = mxGetPr(mxGetField(samples, t, "chi"));

      // Incorporate MF prediction and biases
#pragma omp parallel for
      for(mwSize e = 0; e < numExamples; e++){
	 int u = users[e]-1;
	 int j = items[e]-1;
	 
	 double* cVec = c + numTopicFacs*u;
	 double* dVec = d + numTopicFacs*j;
	 int numTopicFacsTimesNumUsers = numTopicFacs*numUsers;
	 int numTopicFacsTimesNumItems = numTopicFacs*numItems;
	 // Form prediction
	 if(addBase){
	    preds[e] += ddot(&numFacs, a + numFacs*u, &blasStride, 
			     b + numFacs*j, &blasStride);
	 }
	 if(addTopicFactors && (numTopicFacs > 0) && (KU > 0) && (KM > 0)){
	    if(zU != NULL && zM != NULL){
	       // Use provided topics
	       preds[e] += ddot(&numTopicFacs, cVec+
				numTopicFacsTimesNumUsers*(zM[e]-1), &blasStride, 
				dVec+numTopicFacsTimesNumItems*(zU[e]-1), &blasStride);
	    }
	    else{
	       // Integrate over topics
	       preds[e] += integrateFactorVectors(cVec, dVec,
						  logthetaU + u*KU,
						  logthetaM + j*KM, 
						  numTopicFacs, 
						  numTopicFacsTimesNumUsers,
						  numTopicFacsTimesNumItems,
						  KU, KM);
	    }
	 }
	 if(addBiases){
	    preds[e] += xi[u] + chi[j]; 
	 }
      }
   }
   if(numSamples > 1){
      // Average over all sample predictions
      for(mwSize e = 0; e < numExamples; e++)
	 preds[e] /= numSamples;
   }
}
