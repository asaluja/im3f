#ifndef MEX_UTIL_H
#define MEX_UTIL_H

/* 
  MEXUTIL File with utility functions

  ------------------------------------------------------------------------     

  Last revision: 4-July-2010

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

#ifdef NDEBUG
// Fill each entry of double array with val
inline void fillArrayD(double* array, const ptrdiff_t len, const double val){
   for(ptrdiff_t ii = 0; ii < len; ii++)
      array[ii] = val;
}

// Fill each entry of ptrdiff_t array with val
inline void fillArrayI(ptrdiff_t* array, const ptrdiff_t len, const ptrdiff_t val){
   for(ptrdiff_t ii = 0; ii < len; ii++)
      array[ii] = val;
}

// Replace each entry of array with its logarithm
inline void logArrayD(double* array, const ptrdiff_t len){
   for(ptrdiff_t ii = 0; ii < len; ii++)
      array[ii] = log(array[ii]);
}
#endif

// Given a vector of normalized or unnormalized log probabilities 
// of length K, return an integer in {1,...,K} sampled from the 
// discrete distribution.
// Whenever max != -INFINITY, max is treated as maximum unnormalized
// log probability, and normalization of vector is performed.
// When max == -INFINITY, no normalization is performed.
ptrdiff_t sampleDiscreteLogProb(const gsl_rng* rng, const double* logProb, 
			  const ptrdiff_t K, const double max){
   if(K == 1)
      // Only one value exists
      return 1;

   // Normalization factor
   double logSumExp = 0;
   if(!isinf(max)){
      // Vector unnormalized: compute log sum exp of logProb
      for(ptrdiff_t i = 0; i < K; i++){
	 if(!isinf(logProb[i]))
	    logSumExp += exp(logProb[i] - max);
      }
      logSumExp = max + log(logSumExp);
   }

   // Generate U[0,1) random number
   const double rnd = gsl_rng_uniform(rng);
   
   // Compare rnd to cumulative probability sum to sample topic
   double cumsum = isinf(logProb[0]) ? 0 : exp(logProb[0] - logSumExp);
   ptrdiff_t i = 1;
   for(; i < K && rnd > cumsum; i++)
      if(!isinf(logProb[i]))
	 cumsum += exp(logProb[i] - logSumExp);

   return i;
}

#define freeArray(array, len) (for(ptrdiff_t ii = 0; ii < (len); ii++) free((array)[ii]);)

#endif
