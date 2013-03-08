/*
File: sampleCRFDishesMex.c
Date: January 11, 2012
Description: mex version of sampleCRFDishsRealTime.m
Input arugments:
data, model, samp, [isItemTopic iscollapsed]
note that the last argument is a boolean array and should not be passed in 
as two separate arguments. 
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <unistd.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include "mexCommon.h"
#include "mexUtil.h"

//search through uint32_t array to find key
int linearSearchInt (const uint32_t* array, const int key, const size_t size){
  for (int i = 0; i < size; i++)
    if (array[i] == key){ return i; }
  return -1; //if not found
}

//search through double array to find key
int linearSearchDouble(const double* array, const int key, const size_t size){
  for (int i = 0; i < size; i++)
    if (array[i] == key){ return i; }
  return -1; //if not found
}

double updateCRFMu(const double* mu, const double* n, const double resid, const int k, const bool isplus, const mxArray* model){
  double sigmaSqd = (*mxGetPr(mxGetField(model, 0, "sigmaSqd2sigmaSqd0")));  
  return (isplus) ? (mu[k]*(n[k] + sigmaSqd) + resid) / (n[k] +1 + sigmaSqd) : (mu[k]*(n[k] + sigmaSqd) - resid)/(n[k] - 1 + sigmaSqd) ; 
}

double getSigmaStarSqd(const int km, const int ku, const double* nC, const double* nD, const double sigmaSqd, const double invsigmaSqd, const double invsigmaSqd0){
  return sigmaSqd + 1 / (invsigmaSqd0 + invsigmaSqd*nC[km]) + 1 / (invsigmaSqd0 + invsigmaSqd*nD[ku]);
}

double getSigmaDSqd(const int ku, const double* nD, const double invsigmaSqd, const double invsigmaSqd0){
  return 1 / (invsigmaSqd0 + invsigmaSqd*nD[ku]);
}

int sampleDishFull(const uint32_t* examps, const double* resids, const double* mC, const double* muC, const double* muD, const uint32_t* kU, const double* nC, const double* nD, const double betaM, const double c0, const double sigmaSqd, const double invsigmaSqd, const double invsigmaSqd0, const int KM, const int numExamples, const gsl_rng* rng){
  double logmult[KM+1]; //initialize to 0
  for (int kk = 0; kk < KM; kk++ ){
    logmult[kk] = 0; 
    for (int ee_i = 0; ee_i < numExamples; ee_i++ ){
      uint32_t ee = examps[ee_i] - 1;
      double residC = resids[ee] - muD[kU[ee]-1]; 
      double sigmaStarSqd = getSigmaStarSqd(kk, kU[ee]-1, nC, nD, sigmaSqd, invsigmaSqd, invsigmaSqd0); 
      logmult[kk] = logmult[kk] - pow(residC - muC[kk], 2) / (2 * sigmaStarSqd) - log(sigmaStarSqd) / 2; 
    }
  }
  logmult[KM] = 0; 
  for (int ee_i = 0; ee_i < numExamples; ee_i++){
    uint32_t ee = examps[ee_i] - 1;
    double residC = resids[ee] - muD[kU[ee]-1];
    double sigmaDSqd = getSigmaDSqd(kU[ee]-1, nD, invsigmaSqd, invsigmaSqd0);
    //DONGZHEN To do: add sigmaC
    logmult[KM] = logmult[KM] - pow(residC - c0, 2) / (2*(sigmaSqd + sigmaDSqd)) - log(sigmaSqd + sigmaDSqd) / 2;    
  }
  //compute mean, limit max value to counter overflow issues
  double sum = 0; 
  for (int i = 0; i < KM+1; i++ ){ sum += logmult[i]; }
  double average = sum/(KM+1); 
  for (int i = 0; i < KM+1; i++ ){ 
    logmult[i] -= average; //mean shift values
    if (logmult[i] > 30){logmult[i] = 30; } //N.B.: dont hardcode threshold
  }
  double mult[KM+1]; 
  for (int kk = 0; kk < KM; kk++ ){ mult[kk] = mC[kk] * exp(logmult[kk]); }
  mult[KM] = betaM * exp(logmult[KM]); 
  sum = 0; 
  for (int i = 0; i < KM + 1; i++ ){ sum += mult[i]; }
  double multi_norm[KM+1];
  for (int i = 0; i < KM + 1; i++ ){multi_norm[i] = mult[i] / sum; } //need to normalize multinomial
  uint32_t* new_k_sample = (uint32_t*)mxMalloc((KM+1)*sizeof(uint32_t));   //sampling
  gsl_ran_multinomial(rng,(size_t)(KM + 1), 1, multi_norm, new_k_sample); //will return array same length as multi_norm in new_k_sample
  uint32_t new_k = linearSearchInt(new_k_sample, 1, KM+1); 
  if (new_k == -1){ //i.e., we cannot find the sampled idx  
    mexPrintf("Error! Cannot find sampled idx for new dish\n");
    for (int i = 0; i < KM+1; i++ )
      mexPrintf("multi_norm[%d]: %.10f; ", i, multi_norm[i]);
    mexPrintf("\n"); 
  }
  mxFree(new_k_sample); 
  return new_k;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  //setting up inputs
  //first, set up the structs
  const mxArray* data = mxDuplicateArray(prhs[0]);
  const mxArray* model = mxDuplicateArray(prhs[1]);
  const mxArray* samp = mxDuplicateArray(prhs[2]);
  if (data == NULL || model == NULL || samp == NULL){ mexPrintf("Error duplicating inputs\n"); exit(EXIT_FAILURE); }
  const mxLogical* params = mxGetLogicals(prhs[3]); 
  int numExamples = 0; 
  //then, set up the cell array pointers
  mxArray* exampsByUserItem = NULL; 
  mxArray* sampkuM = NULL; 
  mxArray* samptuM = NULL;   
  mxArray* sampnuM = NULL; 
  //then, the double data matrices (as vectors)
  double* c = NULL;
  double* d = NULL; 
  double* muC = NULL; 
  double* muD = NULL; 
  double* mC = NULL; 
  double* nC = NULL; 
  double* nD = NULL; 
  //the scalar values
  double c0, betaM, gammaM;
  c0 = betaM = gammaM = 0; 
  int length_mC, length_nC, length_kM, length_muC; 
  length_mC = length_nC = length_kM = length_muC = 0; 
  //the uint32_t numerical matrices
  uint32_t* kM = NULL; 
  uint32_t* kU = NULL; 
  const double* resids = mxGetPr(mxGetField(samp, 0, "resids")); 
  const double d0 = (*mxGetPr(mxGetField(model, 0, "d0"))); 
  const double sigmaSqd = (*mxGetPr(mxGetField(model, 0, "sigmaSqd"))); 
  const double sigmaSqd0 = (*mxGetPr(mxGetField(model, 0, "sigmaSqd0"))); 
  const double invsigmaSqd = (*mxGetPr(mxGetField(model, 0, "invsigmaSqd")));
  const double invsigmaSqd0 = (*mxGetPr(mxGetField(model, 0, "invsigmaSqd0")));
  if (params[0]){ //isItemTopic == true
    mexPrintf("Sampling dishes for users\n"); 
    numExamples = (*mxGetPr(mxGetField(data, 0, "numUsers"))); //dereference to get value of numUsers
    exampsByUserItem = mxGetField(data, 0, "exampsByUser"); 
    c = mxGetPr(mxGetField(samp, 0, "c")); 
    d = mxGetPr(mxGetField(samp, 0, "d")); 
    sampkuM = mxGetField(samp, 0, "kuM"); 
    samptuM = mxGetField(samp, 0, "tuM"); 
    sampnuM = mxGetField(samp, 0, "nuM"); 
    length_mC = mxGetN(mxGetField(samp, 0, "mC")); 
    mC = (double*)mxMalloc((length_mC)*sizeof(double));
    memcpy(mC, mxGetPr(mxGetField(samp, 0, "mC")), length_mC*sizeof(double)); 
    length_nC = mxGetN(mxGetField(samp, 0, "nC"));
    nC = (double*)mxMalloc((length_nC)*sizeof(double));
    memcpy(nC, mxGetPr(mxGetField(samp, 0, "nC")), length_nC*sizeof(double)); 
    nD = mxGetPr(mxGetField(samp, 0, "nD")); //we don't modify nD, so it's OK to just have it be a pointer
    length_kM = mxGetN(mxGetField(samp, 0, "kM")); 
    kM = (uint32_t*)mxMalloc((length_kM)*sizeof(uint32_t)); 
    memcpy(kM, (uint32_t*)mxGetData(mxGetField(samp, 0, "kM")), length_kM*sizeof(uint32_t));     
    kU = (uint32_t*)mxGetData(mxGetField(samp, 0, "kU")); //not being modified
    length_muC = mxGetN(mxGetField(samp, 0, "muC")); 
    muC = (double*)mxMalloc((length_muC)*sizeof(double)); 
    memcpy(muC, mxGetPr(mxGetField(samp, 0, "muC")), length_muC*sizeof(double));
    muD = mxGetPr(mxGetField(samp, 0, "muD")); 
    c0 = (*mxGetPr(mxGetField(model, 0, "c0")));
    betaM = (*mxGetPr(mxGetField(model, 0, "betaM")));
    gammaM = (*mxGetPr(mxGetField(model, 0, "gammaM")));
  }
  else{ //isItemTopic == false
    mexPrintf("Sampling dishes for items\n");
    numExamples = (*mxGetPr(mxGetField(data, 0, "numItems"))); 
    exampsByUserItem = mxGetField(data, 0, "exampsByItem"); 
    c = mxGetPr(mxGetField(samp, 0, "d")); 
    d = mxGetPr(mxGetField(samp, 0, "c")); 
    sampkuM = mxGetField(samp, 0, "kjU"); 
    samptuM = mxGetField(samp, 0, "tjU"); 
    sampnuM = mxGetField(samp, 0, "njU"); 
    length_mC = mxGetN(mxGetField(samp, 0, "mD")); 
    mC = (double*)mxMalloc((length_mC)*sizeof(double));
    memcpy(mC, mxGetPr(mxGetField(samp, 0, "mD")), length_mC*sizeof(double)); 
    length_nC = mxGetN(mxGetField(samp, 0, "nD"));
    nC = (double*)mxMalloc((length_nC)*sizeof(double));
    memcpy(nC, mxGetPr(mxGetField(samp, 0, "nD")), length_nC*sizeof(double)); 
    nD = mxGetPr(mxGetField(samp, 0, "nC")); 
    length_kM = mxGetN(mxGetField(samp, 0, "kU")); 
    kM = (uint32_t*)mxMalloc((length_kM)*sizeof(uint32_t)); 
    memcpy(kM, (uint32_t*)mxGetData(mxGetField(samp, 0, "kU")), length_kM*sizeof(uint32_t)); 
    kU = (uint32_t*)mxGetData(mxGetField(samp, 0, "kM")); 
    length_muC = mxGetN(mxGetField(samp, 0, "muD")); 
    muC = (double*)mxMalloc((length_muC)*sizeof(double)); 
    memcpy(muC, mxGetPr(mxGetField(samp, 0, "muD")), length_muC*sizeof(double)); 
    muD = mxGetPr(mxGetField(samp, 0, "muC")); 
    c0 = (*mxGetPr(mxGetField(model, 0, "d0")));
    betaM = (*mxGetPr(mxGetField(model, 0, "betaU")));
    gammaM = (*mxGetPr(mxGetField(model, 0, "gammaU")));
  }
  mexPrintf("Initialized values successfully\n"); 
  //outputs
  mxArray* sampkuM_out = mxCreateCellMatrix(numExamples, 1); 
  omp_set_num_threads(MAX_NUM_THREADS);
  gsl_rng** rngs = getRngArray(); //RNG part
  int thread = omp_get_thread_num(); 
  const gsl_rng* rng = rngs[thread]; 
//#pragma omp parallel for
  for (int uu = 0; uu < numExamples; uu++){
    mwSize kuM_size = mxGetN(mxGetCell(sampkuM, uu)); 
    mwSize tuM_size = mxGetN(mxGetCell(samptuM, uu)); 
    mwSize TuM = mxGetN(mxGetCell(sampnuM, uu)); //initialize TuM when going to new user; TuM = nuM_size
    uint32_t* kuM = (uint32_t*)mxMalloc((kuM_size)*sizeof(uint32_t)); //needs to be modified, so allocating on heap
    memcpy(kuM, (uint32_t*)mxGetData(mxGetCell(sampkuM, uu)), kuM_size*sizeof(uint32_t)); //copying onto heap
    uint32_t* tuM = (uint32_t*)mxMalloc(tuM_size*sizeof(uint32_t)); 
    memcpy(tuM, (uint32_t*)mxGetData(mxGetCell(samptuM, uu)), tuM_size*sizeof(uint32_t));
    uint32_t* examps = (uint32_t*)mxGetData(mxGetCell(exampsByUserItem, uu)); 
    int numExamples_uu = mxGetN(mxGetCell(exampsByUserItem, uu)); //size of examps
    uint32_t** tuM_cell = (uint32_t**)mxMalloc(TuM*sizeof(uint32_t*)); 
    uint32_t tuM_cell_size[TuM];
    for (int i = 0; i < TuM; i++ ){
      tuM_cell[i] = NULL;
      tuM_cell_size[i] = 0; 
    }
    if (tuM_size != numExamples_uu){ mexPrintf("Error: size of tuM matrix for user %d is %d; it should equal numExamples, which is %d\n", uu, tuM_size, numExamples_uu); exit(EXIT_FAILURE); }
    for (mwSize ee_i = 0; ee_i < numExamples_uu; ee_i++){ //numExamples_uu == tuM_size; assembling all the examples for this user sorted by table in a cell array
      if (tuM[ee_i] > TuM){ mexPrintf("Error: table number for user %d, example %d, exceeds number of tables!\n", uu, ee_i); }
      uint32_t* tuM_cell_table = tuM_cell[tuM[ee_i]-1]; 
      if (tuM_cell_table == NULL){ //i.e., the entry is empty thus far
	tuM_cell_table = (uint32_t*)mxMalloc(sizeof(uint32_t));
	tuM_cell_table[0] = examps[ee_i]; 
	tuM_cell[tuM[ee_i]-1] = tuM_cell_table; 
	tuM_cell_size[tuM[ee_i]-1] = 1; 
      }
      else {
	int orig_size = tuM_cell_size[tuM[ee_i]-1]; 
	uint32_t* tuM_cell_expanded = (uint32_t*)mxRealloc(tuM_cell_table, (orig_size+1)*sizeof(uint32_t));
	if (tuM_cell_expanded){ 
	  tuM_cell_expanded[orig_size] = examps[ee_i];
	  tuM_cell[tuM[ee_i]-1] = tuM_cell_expanded; 
	  tuM_cell_size[tuM[ee_i]-1] += 1; 
	}
	else { mexPrintf("Could not enlarge tuM cell array for user %d, example %d\n", uu, ee_i); }
      }
    }
    //sample a new dish for each table
    for (mwSize tt = 0; tt < TuM; tt++ ){
      uint32_t* examp_tuM = tuM_cell[tt];
      int table_size = tuM_cell_size[tt]; 
      if (table_size > 0){
	int old_k = kuM[tt] - 1;
	//update global dish sufficient stats immediately
	mC[old_k] -= 1; 
	if (mC[old_k] < 0){ mC[old_k] = 0; }
	for (mwSize ee_i = 0; ee_i < table_size; ee_i++){ //loop through all examples assigned to table
	  uint32_t ee = examp_tuM[ee_i] - 1;
	  double residC = (params[1]) ? resids[ee] - muD[kU[ee]-1] : resids[ee] - d[kU[ee]-1]; 	  
	  muC[old_k] = updateCRFMu(muC, nC, residC, old_k, false, model); //remove current rating from global sufficient stats
	  //NOTE! this is a hack
	  if (fabs(muC[old_k]) > 30){ int sign = (muC[old_k] > 0) - (muC[old_k] < 0); muC[old_k] = sign*30; }
	  nC[old_k] -= 1; 
	}
	int new_k = sampleDishFull(examp_tuM, resids, mC, muC, muD, kU, nC, nD, betaM, c0, sigmaSqd, invsigmaSqd, invsigmaSqd0, length_mC, table_size, rng);	
	//skipping if condition that checks if length(new_k) > 1
	int empty_dish = linearSearchDouble(nC, 0, length_nC); //see what table we selected      
	if (empty_dish > -1){ new_k = empty_dish; } //if empty_dish is found, then set new_k to it
	kuM[tt] = new_k + 1; //+1 to be consistent with matlab notations
	if (new_k + 1 > length_mC){ //length_mC tells us number of active dishes right now
	  double* mC_expanded = (double*)mxRealloc(mC, (length_mC+1)*sizeof(double)); 
	  if (mC_expanded) { mC = mC_expanded; }
	  length_mC++; 
	  mC[new_k] = 0; //new dish
	} 
	mC[new_k] += 1; 
	if (new_k + 1 > length_nC){
	  double* muC_expanded = (double*)mxRealloc(muC, (length_muC+1)*sizeof(double));
	  if (muC_expanded) { muC = muC_expanded; }
	  double* nC_expanded = (double*)mxRealloc(nC, (length_nC+1)*sizeof(double));
	  if (nC_expanded) { nC = nC_expanded; }
	  length_nC++;
	  length_muC++;
	  muC[new_k] = d0 * sigmaSqd / sigmaSqd0;
	  nC[new_k] = 0;
	}
	for (mwSize ee_i = 0; ee_i < table_size; ee_i++){
	  uint32_t ee = examp_tuM[ee_i] - 1;	  
	  kM[ee] = new_k+1; 
	  double residC = (params[1]) ? resids[ee] - muD[kU[ee]-1] : resids[ee] - d[kU[ee]-1];
	  muC[new_k] = updateCRFMu(muC, nC, residC, new_k, true, model); 
	  //NOTE: this is a hack!
	  if (fabs(muC[new_k]) > 30){ int sign = (muC[new_k] > 0) - (muC[new_k] < 0); muC[new_k] = sign*30; }
	  nC[new_k] += 1; 
	}
      }
    } //close loop that goes through the tables for a given user
    mxArray* kuM_out; 
    kuM_out = mxCreateNumericMatrix(1, TuM, mxUINT32_CLASS, mxREAL); 
    if (kuM_out){
      mxSetPr(kuM_out, kuM); 
      mxSetCell(sampkuM_out, uu, kuM_out); 
    }
  } //close loop over all users
  //set up outputs for remaining variables
  mxArray* mC_out = mxCreateDoubleMatrix(1, length_mC, mxREAL); 
  mxArray* nC_out = mxCreateDoubleMatrix(1, length_nC, mxREAL); 
  mxArray* muC_out = mxCreateDoubleMatrix(1, length_muC, mxREAL); 
  mxArray* kM_out = mxCreateNumericMatrix(1, length_kM, mxUINT32_CLASS, mxREAL);   
  mxSetPr(mC_out, mC); 
  mxSetPr(nC_out, nC);
  mxSetPr(muC_out, muC);
  mxSetPr(kM_out, kM); 
  plhs[0] = sampkuM_out;
  plhs[1] = mC_out;
  plhs[2] = nC_out; 
  plhs[3] = muC_out; 
  plhs[4] = kM_out; 
  mexPrintf("Finished sampling dishes\n"); 
}
