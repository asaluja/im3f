/* 
  ACCUMARRAYMEX Perform 3 argument accumarray functionality for up to 2-D outputs
  
  Usage: 
     [a] = accumarrayMex(indices, vals, sizeArray)
 
  Inputs:
     indices - Single column of uint32 indices or pair of columns of uint32 indices
     vals - Single double value or array of double values to be accumulated
     sizeArray - Target size of accumulation matrix
 
  Outputs: 
     a - Array with size sizeArray created by accumulating elements of 
     vals using the indices in indices (see ACCUMARRAY)
  
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
#include "mexCommon.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   // Extract input information
   uint32_t* indices = (uint32_t*) mxGetData(prhs[0]);
   const ptrdiff_t numIndRows = mxGetM(prhs[0]);
   const ptrdiff_t numIndCols = mxGetN(prhs[0]);
   double* vals = mxGetPr(prhs[1]);
   const ptrdiff_t numVals = mxGetM(prhs[1]);
   double* sizeArray = mxGetPr(prhs[2]);
   const ptrdiff_t numAccRows = sizeArray[0]+.5;

   // Create output array
   // New array auto-initialized to zeros
   plhs[0] = mxCreateDoubleMatrix(numAccRows, sizeArray[1] + .5, mxREAL);
   double* accums = mxGetPr(plhs[0]);

   // Form accumulated values
   if(numIndCols > 1){
      // Two column case
      for(ptrdiff_t e = 0; e < numIndRows; e++){
         ptrdiff_t rowInd = indices[e] - 1;
         ptrdiff_t colInd = indices[e+numIndRows] - 1;
         if(numVals > 1)
            accums[colInd*numAccRows + rowInd] += vals[e];
         else
            accums[colInd*numAccRows + rowInd] += vals[0];
      }
   }
   else{
      // One column case
      if(numVals > 1){
         for(ptrdiff_t e = 0; e < numIndRows; e++)
            accums[indices[e] - 1] += vals[e];
      }
      else{
         double val = vals[0];
         for(ptrdiff_t e = 0; e < numIndRows; e++)
            accums[indices[e] - 1] += val;
      }
   }
}
