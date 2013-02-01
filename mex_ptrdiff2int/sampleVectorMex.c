/* 
  SAMPLEVECTORMEX Sample a vector of uint32 categorical variables.
  
  Usage: 
     [vec] = sampleVectorMex(p, cols)
 
  Inputs:
     p - Each column of p is a distribution over categorical variables
     cols - uint32 variables indexing columns of p
 
  Outputs: 
     vec - The ith variable is sampled from p(:,cols[i]).
  
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

#include <omp.h>
#include <stdint.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "mex.h"
#include "mexCommon.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   omp_set_num_threads(MAX_NUM_THREADS);

   // Array of static random number generators
   gsl_rng** rngs = getRngArray();

   // Extract input information
   const double* p = mxGetPr(prhs[0]); // K x numDists
   const ptrdiff_t K = mxGetM(prhs[0]); // Number of categories
   const uint32_t* cols = (uint32_t*)mxGetData(prhs[1]); // vecLen x 1
   const ptrdiff_t vecLen = mxGetM(prhs[1]);

   // Prepare output
   // New array auto-initialized to zeros
   plhs[0] = mxCreateNumericMatrix(vecLen, 1, mxUINT32_CLASS, mxREAL);
   uint32_t* vec = (uint32_t*)mxGetData(plhs[0]);

#pragma omp parallel for
   for(mwSize e = 0; e < vecLen; e++){
      // Find correct probability vector
      const double* pVec = p + (K*(cols[e]-1));

      // Generate U(0,1) random number
      const double rnd = gsl_rng_uniform(rngs[omp_get_thread_num()]);

      // Compare rnd to cumulative probability sum to sample topic
      double cumsum = pVec[0];
      uint32_t i = 1;
      for(; (i < K) && (rnd > cumsum); i++)
	 cumsum += pVec[i];
      vec[e] = i;
   }
}
