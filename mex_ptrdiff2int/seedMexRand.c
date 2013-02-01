/* 
  SEEDMEXRAND Seed random number generators used by MEX files.
  
  Usage: 
     seedMexRand(seed)
 
  Inputs:
     seed - Number used to seed random number generators.
  
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
#include "mex.h"
#include "mexCommon.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   // Extract input information
   const unsigned long seed = *mxGetPr(prhs[0]) + .5;

   mexPrintf("Running seedMexRand with seed = %lu\n", seed);
   gsl_rng** rngs = getRngArray();
   for(unsigned long r = 0; r < MAX_NUM_THREADS; r++){
      gsl_rng_set(rngs[r], 2*(r+seed)+1);
   }
}
