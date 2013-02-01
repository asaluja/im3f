/* 
  SAMPLETOPICPARAMS Samples topic parameter distributions
   
  Usage: 
     m3f_tib_sampleTopicParams(data, model, samp, zU, zM)

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - model info structure 
     samp - current instantation of model parameters for gibbs sampling
     zU,zM - current sampled user/item latent topics

  Outputs:
     This function modifies the 'samp' structure IN PLACE, 
     resampling item and user topic distributions.

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

// Sample topic parameters
// Function written from perspective of sampling user topics parameters
// Switch roles of user-item inputs to sample item topics parameters
void sampleTopicParams(const mxArray* exampsByUser, ptrdiff_t KU, ptrdiff_t numUsers,
                       double alpha, double* logthetaU, uint32_t* zU){

   // Array of static random number generators
   gsl_rng** rngs = getRngArray();

   // Prior term for Dirichlet
   const double ratio = alpha/KU;

   // Allocate memory for storing topic counts
   double* counts[MAX_NUM_THREADS];
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++)
      counts[thread] = mxMalloc(KU*sizeof(**counts));

#pragma omp parallel for
   for(ptrdiff_t u = 0; u < numUsers; u++){
      ptrdiff_t thread = omp_get_thread_num();
      // Initialize to prior term
      for(ptrdiff_t i = 0; i < KU; i++)
         counts[thread][i] = ratio;

      // Iterate over user's examples computing sufficient stats
      mxArray* exampsArray = mxGetCell(exampsByUser, u);
      mwSize len = mxGetN(exampsArray);
      uint32_t* examps = (uint32_t*) mxGetData(exampsArray);
      for(ptrdiff_t j = 0; j < len; j++)
         counts[thread][zU[examps[j]-1]-1]++;

      // Sample new topic parameters
      double* logthetaPtr = logthetaU + u*KU;
      gsl_ran_dirichlet(rngs[omp_get_thread_num()], KU, counts[thread], 
			logthetaPtr);
      // Take logs
      for(ptrdiff_t i = 0; i < KU; i++)
         logthetaPtr[i] = log(logthetaPtr[i]);
   }
   // Clean up
   for(ptrdiff_t thread = 0; thread < MAX_NUM_THREADS; thread++)
      mxFree(counts[thread]);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running sampleTopicParams\n");

   omp_set_num_threads(MAX_NUM_THREADS);

   // Extract input information
   const mxArray* data = prhs[0];
   const mxArray* exampsByUser = mxGetField(data, 0, "exampsByUser");
   const mxArray* exampsByItem = mxGetField(data, 0, "exampsByItem");
   const mxArray* model = prhs[1];
   ptrdiff_t KU = (*mxGetPr(mxGetField(model, 0, "KU"))) + .5;
   ptrdiff_t KM = (*mxGetPr(mxGetField(model, 0, "KM"))) + .5;
   ptrdiff_t numUsers = (*mxGetPr(mxGetField(model, 0, "numUsers"))) + .5;
   ptrdiff_t numItems = (*mxGetPr(mxGetField(model, 0, "numItems"))) + .5;
   double alpha = (*mxGetPr(mxGetField(model, 0, "alpha")));
   const mxArray* samp = prhs[2];
   double* logthetaU = mxGetPr(mxGetField(samp, 0, "logthetaU")); // KU x numUsers
   double* logthetaM = mxGetPr(mxGetField(samp, 0, "logthetaM")); // KM x numItems
   uint32_t* zU = (uint32_t*)mxGetData(prhs[3]);
   uint32_t* zM = (uint32_t*)mxGetData(prhs[4]);
   mxLogical* sampParams = NULL;
   if(nrhs > 5){
      sampParams = mxGetLogicals(prhs[5]);
   }

   // Sample user topic parameters
   if((KU > 1) && ((sampParams == NULL) || sampParams[0])){
      sampleTopicParams(exampsByUser, KU, numUsers, alpha, logthetaU, zU);
   }
   // Sample item topic parameters
   if((KM > 1) && ((sampParams == NULL) || sampParams[1])){
      sampleTopicParams(exampsByItem, KM, numItems, alpha, logthetaM, zM);
   }
}
