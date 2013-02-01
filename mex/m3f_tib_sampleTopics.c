/* 
  M3F_TIB_SAMPLETOPICS Gibbs samples the topic parameters and topics.
   
  Usage: 
     m3f_tib_sampleTopics(data, model, samp, zU, zM, resids)

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - model info structure 
     samp - current instantation of model parameters for gibbs sampling
     zU,zM - current sampled user/item latent topics
     resids - differences between true ratings and base rating predictions
     (see m3f_tib_gibbs)

  Outputs:
     This function modifies the 'zU' and 'zM' vectors IN PLACE,
     resampling the item and user topic for each rating.

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
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "mex.h"
#include "mexCommon.h"
#include "mexUtil.h"

// Sample topics
// Function written from perspective of sampling user topics
// Switch roles of user-item inputs to sample item topics
void sampleTopics(uint32_t* users, uint32_t* items, ptrdiff_t KU, ptrdiff_t KM,
                  double twoSigmaSqd, double* logthetaU, double* c,
                  double* d, uint32_t* zU, uint32_t* zM, double* resids,
                  mwSize numExamples){
   // Array of static random number generators
   gsl_rng** rngs = getRngArray();

   // Allocate memory for log probabilities
   double* logProb[MAX_NUM_THREADS];
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++)
     logProb[thread] = mxMalloc(KU*sizeof(**logProb));

#pragma omp parallel for
   for(mwSize e = 0; e < numExamples; e++){
      ptrdiff_t thread = omp_get_thread_num();
      // Compute logthetaU - log likelihood term
      ptrdiff_t u = users[e]-1;
      double* logthetaPtr = logthetaU + u*KU;
      double* dPtr = d + (items[e]-1)*KU;
      double residMinusC;
      if(KM > 0)
         residMinusC = resids[e] - c[u*KM + (zM[e]-1)];
      else
         residMinusC = resids[e];
      double max = -INFINITY;
      for(ptrdiff_t i = 0; i < KU; i++){
         double err = residMinusC - dPtr[i];
         logProb[thread][i] = logthetaPtr[i] - err*err/twoSigmaSqd;
         if(logProb[thread][i] > max)
            max = logProb[thread][i];
      }

      zU[e] = sampleDiscreteLogProb(rngs[thread],
				    logProb[thread], KU, max);
   }
   // Clean up
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++)
      mxFree(logProb[thread]);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running m3f_tib_sampleTopics\n");

   omp_set_num_threads(MAX_NUM_THREADS);

   // Extract input information
   const mxArray* data = prhs[0];
   uint32_t* users = (uint32_t*)mxGetData(mxGetField(data, 0, "users"));
   uint32_t* items = (uint32_t*)mxGetData(mxGetField(data, 0, "items"));
   const mxArray* model = prhs[1];
   ptrdiff_t KU = (*mxGetPr(mxGetField(model, 0, "KU"))) + .5;
   ptrdiff_t KM = (*mxGetPr(mxGetField(model, 0, "KM"))) + .5;
   double twoSigmaSqd = (*mxGetPr(mxGetField(model, 0, "sigmaSqd")))*2;
   const mxArray* samp = prhs[2];
   double* logthetaU = mxGetPr(mxGetField(samp, 0, "logthetaU")); // KU x numUsers
   double* logthetaM = mxGetPr(mxGetField(samp, 0, "logthetaM")); // KM x numItems
   double* c = mxGetPr(mxGetField(samp, 0, "c")); // KM x numUsers
   double* d = mxGetPr(mxGetField(samp, 0, "d")); // KU x numItems
   uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
   uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
   double* resids = mxGetPr(prhs[5]);
   mwSize numExamples = mxGetM(prhs[5]);

   // Sample user topics
   if(KU > 1){
      sampleTopics(users, items, KU, KM, twoSigmaSqd, logthetaU, 
		   c, d, zU, zM, resids, numExamples);
   }

   // Sample item topics
   if(KM > 1){
      sampleTopics(items, users, KM, KU, twoSigmaSqd, logthetaM, 
		   d, c, zM, zU, resids, numExamples);
   }
}
