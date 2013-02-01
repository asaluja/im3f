/* 
  M3F_TIF_SAMPLEBIASES samples xi and chi for TIF model from posterior

  Usage: 
     m3f_tif_sampleBiases(data, model, samp, resids)
     m3f_tif_sampleBiases(data, model, samp, resids, [sampUserParams, sampItemParams])

  Inputs: 
     data - Dyadic data structure (see loadDyadicData)
     model - m3f_tif structure (see m3f_tif_initModel)
     samp - Current Gibbs sample of model variables
     resids - differences between true ratings and base rating predictions
     sampleUserParams - OPTIONAL: if false, user params are not sampled
     sampleItemParams - OPTIONAL: if false, item params are not sampled

  Outputs:
     This function modifies the "samp" structure input IN PLACE, 
     replacing the 'xi','chi', 'c' and 'd' samples.

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

// Sample biases
// Function written from perspective of sampling xi biases
// Switch roles of user-item inputs to sample chi biases
void sampleBiases(uint32_t* users, uint32_t* items,  const mxArray* exampsByUser,
                  int numUsers, double invSigmaSqd, double invSigmaSqd0, 
		  double xi0, double* xi, double* chi, double* resids){

   // Array of static random number generators
   gsl_rng** rngs = getRngArray();

   // Prior term for biases
   const double ratio = xi0*invSigmaSqd0;

#pragma omp parallel for
   for(int u = 0; u < numUsers; u++){
      // Iterate over user's examples computing sufficient stats
      mxArray* exampsArray = mxGetCell(exampsByUser, u);
      mwSize len = mxGetN(exampsArray);
      uint32_t* examps = (uint32_t*) mxGetData(exampsArray);
      double ss = 0;
      for(int j = 0; j < len; j++){
         uint32_t e = examps[j]-1;
	 ss += (resids[e] - chi[items[e]-1]);
      }

      // Sample new bias values using sufficient stats
      double variance = 1.0/(invSigmaSqd0 + len*invSigmaSqd);
      xi[u] = (ratio + ss*invSigmaSqd)*variance +
	 gsl_ran_gaussian(rngs[omp_get_thread_num()], sqrt(variance));
   }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   mexPrintf("Running m3f_tif_sampleBiases\n");

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
   double invSigmaSqd = 1.0/(*mxGetPr(mxGetField(model, 0, "sigmaSqd")));
   double invSigmaSqd0 = 1.0/(*mxGetPr(mxGetField(model, 0, "sigmaSqd0")));
   double xi0 = (*mxGetPr(mxGetField(model, 0, "xi0")));
   double chi0 = (*mxGetPr(mxGetField(model, 0, "chi0")));
   const mxArray* samp = prhs[2];
   double* xi = mxGetPr(mxGetField(samp, 0, "xi")); 
   double* chi = mxGetPr(mxGetField(samp, 0, "chi")); 
   double* resids = mxGetPr(prhs[3]);
   mxLogical* sampParams = NULL;
   if(nrhs > 4){
      sampParams = mxGetLogicals(prhs[4]);
   }

   // Sample xi biases
   if((sampParams == NULL) || sampParams[0]){
      sampleBiases(users, items,  exampsByUser, numUsers,
		   invSigmaSqd, invSigmaSqd0, xi0, xi, chi,
		   resids);
   }

   // Sample chi biases
   if((sampParams == NULL) || sampParams[1]){
      sampleBiases(items, users, exampsByItem, numItems,
		   invSigmaSqd, invSigmaSqd0, chi0, chi, xi,
		   resids);
   }
}
