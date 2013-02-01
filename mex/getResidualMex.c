/*
File: getResidualMex.c
Date: January 4, 2012
Description: mex version of getResidual.c
Input arugments:
users, items, values, samples
 */

#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include "mexCommon.h"
#include "mexUtil.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  //may want to add some error checking here for number of args, perhaps type
  if (nlhs > 1){ mexErrMsgTxt("Too many output arguments");}
  mexPrintf("Getting residuals\n");
  omp_set_num_threads(MAX_NUM_THREADS);
  //extract input information
  uint32_t* users = (uint32_t*)mxGetData(prhs[0]);
  mwSize numExamples = mxGetM(prhs[0]);
  uint32_t* items = (uint32_t*)mxGetData(prhs[1]);
  double* vals = (double*)mxGetData(prhs[2]);
  const mxArray* samples = prhs[3];
  double* a = mxGetPr(mxGetField(samples, 0, "a")); 
  double* b = mxGetPr(mxGetField(samples, 0, "b")); 
  const double chi = *mxGetPr(mxGetField(samples, 0, "chi")); //I think this should stay the same no matter what right
  
  ptrdiff_t numFacs = mxGetM(mxGetField(samples, 0, "a")); 
  //set up output
  plhs[0] = mxCreateDoubleMatrix(1, numExamples, mxREAL);
  double* resids = mxGetPr(plhs[0]);
  ptrdiff_t  blasStride = 1;
#pragma omp parallel for
  for (mwSize ee = 0; ee < numExamples; ee++){
    double* aVec = a + numFacs*(users[ee]-1); //updating the pointer
    double* bVec = b + numFacs*(items[ee]-1); 
    resids[ee] = vals[ee] - chi - ddot(&numFacs, aVec, &blasStride, bVec, &blasStride);     
  }
  mexPrintf("Finished getting residuals\n"); 
}
