/*
File: sampleCRFTablesMex.c
Date: January 4, 2012
Description: mex version of sampleCRFTablesRealRealTime.m
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

int compare_uint32 (const void* a, const void* b){
  if ( *(uint32_t*)a > *(uint32_t*)b ) return 1;
  if ( *(uint32_t*)a == *(uint32_t*)b ) return 0; 
  if ( *(uint32_t*)a < *(uint32_t*)b ) return -1;
}
//search through uint32_t array to find key
int linearSearch (const uint32_t* array, const int key, const size_t size){
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
    //DONGZHEN's To do: add sigmaC
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
  uint32_t new_k = linearSearch(new_k_sample, 1, KM+1); 
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
  mxArray* sampnuM = NULL; 
  mxArray* samptuM = NULL;   
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
  int length_mC, length_nC, length_tM, length_kM, length_muC; 
  length_mC = length_nC = length_tM = length_kM = length_muC = 0; 
  //the uint32_t numerical matrices
  uint32_t* tM = NULL; 
  uint32_t* kM = NULL; 
  uint32_t* kU = NULL; 
  const double* resids = mxGetPr(mxGetField(samp, 0, "resids")); 
  const double d0 = (*mxGetPr(mxGetField(model, 0, "d0"))); 
  if (params[0]){ //isItemTopic == true
    mexPrintf("Sampling tables for users\n"); 
    numExamples = (*mxGetPr(mxGetField(data, 0, "numUsers"))); //dereference to get value of numUsers
    exampsByUserItem = mxGetField(data, 0, "exampsByUser"); 
    c = mxGetPr(mxGetField(samp, 0, "c")); 
    d = mxGetPr(mxGetField(samp, 0, "d")); 
    sampkuM = mxGetField(samp, 0, "kuM"); 
    sampnuM = mxGetField(samp, 0, "nuM");
    samptuM = mxGetField(samp, 0, "tuM"); 
    length_mC = mxGetN(mxGetField(samp, 0, "mC")); 
    mC = (double*)mxMalloc((length_mC)*sizeof(double));
    memcpy(mC, mxGetPr(mxGetField(samp, 0, "mC")), length_mC*sizeof(double)); 
    length_nC = mxGetN(mxGetField(samp, 0, "nC"));
    nC = (double*)mxMalloc((length_nC)*sizeof(double));
    memcpy(nC, mxGetPr(mxGetField(samp, 0, "nC")), length_nC*sizeof(double)); 
    nD = mxGetPr(mxGetField(samp, 0, "nD")); //we don't modify nD, so it's OK to just have it be a pointer
    length_tM = mxGetN(mxGetField(samp, 0, "tM"));
    tM = (uint32_t*)mxMalloc((length_tM)*sizeof(uint32_t)); 
    memcpy(tM, (uint32_t*)mxGetData(mxGetField(samp, 0, "tM")), length_tM*sizeof(uint32_t)); 
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
    mexPrintf("Sampling tables for items\n");
    numExamples = (*mxGetPr(mxGetField(data, 0, "numItems"))); 
    exampsByUserItem = mxGetField(data, 0, "exampsByItem"); 
    c = mxGetPr(mxGetField(samp, 0, "d")); 
    d = mxGetPr(mxGetField(samp, 0, "c")); 
    sampkuM = mxGetField(samp, 0, "kjU"); 
    sampnuM = mxGetField(samp, 0, "njU"); 
    samptuM = mxGetField(samp, 0, "tjU"); 
    length_mC = mxGetN(mxGetField(samp, 0, "mD")); 
    mC = (double*)mxMalloc((length_mC)*sizeof(double));
    memcpy(mC, mxGetPr(mxGetField(samp, 0, "mD")), length_mC*sizeof(double)); 
    length_nC = mxGetN(mxGetField(samp, 0, "nD"));
    nC = (double*)mxMalloc((length_nC)*sizeof(double));
    memcpy(nC, mxGetPr(mxGetField(samp, 0, "nD")), length_nC*sizeof(double)); 
    nD = mxGetPr(mxGetField(samp, 0, "nC")); 
    length_tM = mxGetN(mxGetField(samp, 0, "tU")); 
    tM = (uint32_t*)mxMalloc((length_tM)*sizeof(uint32_t)); 
    memcpy(tM, (uint32_t*)mxGetData(mxGetField(samp, 0, "tU")), length_tM*sizeof(uint32_t)); 
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
  //outputs (some of them, the ones that get written after every user are declared here)
  mxArray* sampkuM_out = mxCreateCellMatrix(numExamples, 1); 
  mxArray* sampnuM_out = mxCreateCellMatrix(numExamples, 1); 
  mxArray* samptuM_out = mxCreateCellMatrix(numExamples, 1); 

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
    uint32_t* nuM = (uint32_t*)mxMalloc(TuM*sizeof(uint32_t)); 
    memcpy(nuM, (uint32_t*)mxGetData(mxGetCell(sampnuM, uu)), TuM*sizeof(uint32_t));
    uint32_t* tuM = (uint32_t*)mxMalloc(tuM_size*sizeof(uint32_t)); 
    memcpy(tuM, (uint32_t*)mxGetData(mxGetCell(samptuM, uu)), tuM_size*sizeof(uint32_t));
    uint32_t* examps = (uint32_t*)mxGetData(mxGetCell(exampsByUserItem, uu)); 
    int numExamples_uu = mxGetN(mxGetCell(exampsByUserItem, uu)); //size of examps
    bool last_example_got_new_table = true; 
    for (mwSize ee_i = 0; ee_i < numExamples_uu; ee_i++){
      uint32_t new_table_idx; 
      if (last_example_got_new_table){ 	//find an old empty table in advance
	new_table_idx = linearSearch(nuM, 0, TuM); 
	if (new_table_idx == -1){ new_table_idx = TuM; } //-1 means key not found, i.e., there is no old empty table
      }
      uint32_t ee = examps[ee_i]-1; //-1 because of the difference in C and matlab indexing
      uint32_t old_t = tuM[ee_i]-1; //original table
      double residC = (params[1]) ? resids[ee] - muD[kU[ee]-1] : resids[ee] - d[kU[ee]-1]; //based on collapsed or non-collapsed
      nuM[old_t]--; //remove current rating from table
      if (nuM[old_t] == 0){ mC[kuM[old_t]-1] -= 1; }//mexPrintf("Removing table for user %d, example %d, indexed at %d; dish count:%.5f-->%.5f\n", uu, ee_i, old_t, mC[kuM[old_t]-1]+1, mC[kuM[old_t]-1]);} //remove table as needed
      int old_k = kuM[old_t]-1; //old dish
      muC[old_k] = updateCRFMu(muC, nC, residC, old_k, false, model); //remove current rating from global sufficient stats
      //NOTE: this is a hack!!
      if (fabs(muC[old_k]) > 30){ int sign = (muC[old_k] > 0) - (muC[old_k] < 0); muC[old_k] = sign*30; }
      //if (fabs(muC[old_k]) < 0.0000000001){ int sign = (muC[old_k] > 0) - (muC[old_k] < 0); muC[old_k] = sign*0.0000000001; }
      nC[old_k] -= 1; //cannot use -- because nC is a double array
      //assemble multinomial probability vector below
      double mult[TuM+1]; 
      const double sigmaSqd = (*mxGetPr(mxGetField(model, 0, "sigmaSqd"))); 
      const double invsigmaSqd = (*mxGetPr(mxGetField(model, 0, "invsigmaSqd"))); 
      const double invsigmaSqd0 = (*mxGetPr(mxGetField(model, 0, "invsigmaSqd0"))); 
      const double sigmaSqd0 = (*mxGetPr(mxGetField(model, 0, "sigmaSqd0"))); 
      for (mwSize tt = 0; tt < TuM; tt++ ){ //go through each table and assign the probability of selecting that table (likelihood x prior)
	const double sigmaStarSqd = getSigmaStarSqd(kuM[tt]-1, kU[ee]-1, nC, nD, sigmaSqd, invsigmaSqd, invsigmaSqd0); 
	mult[tt] = nuM[tt] * exp(-pow(residC - muC[kuM[tt]-1], 2)/(2*sigmaStarSqd)) / (pow(sigmaStarSqd, 0.5)); 
	if (isnan(mult[tt])){
	    mexPrintf("likelihood for table %d and example %d is NaN!\n", tt, ee);
	    mexPrintf("SigmaStarSqd is %.5f, dish assignment is %d, muC is %.5f, residC is %.5f\n", sigmaStarSqd, kuM[tt]-1, muC[kuM[tt]-1], residC);
	  }
      }
      //iterate through all existing dishes to get the probability of assigning to a new dish
      const double sigmaDSqd = getSigmaDSqd(kU[ee]-1, nD, invsigmaSqd, invsigmaSqd0); 
      //TODO (from Dongzhen): change this sigmaC (=sigma0)
      mult[TuM] = betaM * exp(-pow(residC - c0, 2)/(2*(sigmaSqd + sigmaDSqd))) / (pow(sigmaSqd + sigmaDSqd, 0.5)); 
      if (isnan(mult[TuM])){
	mexPrintf("Likelihood for sampling new table NaN!\n"); 
      }
      double divid = betaM; 
      for (mwSize kk = 0; kk < length_mC; kk++ ){ //when sampling a new dish, we are fully Bayesian and go through all dishes
	const double sigmaStarSqd = getSigmaStarSqd(kk, kU[ee]-1, nC, nD, sigmaSqd, invsigmaSqd, invsigmaSqd0); 
	mult[TuM] += mC[kk] * exp(-pow(residC - muC[kk], 2)/(2*sigmaStarSqd)) / (pow(sigmaStarSqd, 0.5)); 
	divid += mC[kk]; 
	if (isnan(mult[TuM])){
	    mexPrintf("likelihood for unused table %d and example %d is NaN!\n", TuM, ee);
	    mexPrintf("SigmaStarSqd is %.5f, dish assignment is %d, muC is %.5f, mC is %.5f, residC is %.5f\n", sigmaStarSqd, kk, muC[kk], mC[kk], residC);
	}
      }
      mult[TuM] = (gammaM * mult[TuM]) / divid; //probability of sampling new table
      if (isnan(mult[TuM])){
	mexPrintf("After dividing by divid and multiplying by gamma, mult[TuM] becomes NaN!\n");
	mexPrintf("GammaM is %.5f, divid is %.5f\n", gammaM, divid); 
      }
      //for (int i = 0; i < TuM + 1; i++ ){mexPrintf("mult[%d]: %.5f; ", i, mult[i]);}
      //mexPrintf("\n");
      double sum = 0; 
      for (int i = 0; i < TuM + 1; i++ ){ sum += mult[i]; }
      if (!(sum > 0)){ mexPrintf("Normalizer is 0! User %d, example %d\n", uu, ee); }
      double multi_norm[TuM+1]; 
      for (int i = 0; i < TuM + 1; i++ ){multi_norm[i] = mult[i] / sum; } //normalized multinorm
      //sample new table assignments from multinomial, update nuM, tuM, kuM
      uint32_t* new_t_sample = (uint32_t*)mxMalloc((TuM+1)*sizeof(uint32_t)); 
      gsl_ran_multinomial(rng, (size_t)(TuM + 1), 1, multi_norm, new_t_sample); //will return array same length as multi_norm in new_t
      uint32_t new_t = linearSearch(new_t_sample, 1, TuM+1); //see what table we selected      
      if (new_t == -1){
	mexPrintf("Error! Cannot find sampled idx for new table for user %d and example %d\n", uu, ee);
	for (int i = 0; i < TuM+1; i++ )
	  mexPrintf("multi_norm[%d]: %.5f; ", i, multi_norm[i]);
	mexPrintf("\n"); 
	double mult_unif[TuM+1];
	for (int i = 0; i < TuM + 1; i++ ){ mult_unif[i] = ((double)1.0) / (TuM + 1); }
	gsl_ran_multinomial(rng, (size_t)(TuM+1), 1, mult_unif, new_t_sample); 
	new_t = linearSearch(new_t_sample, 1, TuM+1); 
      }
      mxFree(new_t_sample); 
      //two if conditions in original matlab code left out since it seems they're never hit
      tuM[ee_i] = new_t + 1; //immediately update the table assignment
      if (new_t + 1 > TuM){ //i.e., we have selected the new table
	if (new_table_idx == TuM){ //if we don't have a previous 0 entry, need to expand nuM, kuM to accommodate new table statistics
	  uint32_t* nuM_expanded = (uint32_t*)mxRealloc(nuM, (TuM+1)*sizeof(uint32_t));
	  if (nuM_expanded){ nuM = nuM_expanded; }
	  uint32_t* kuM_expanded = (uint32_t*)mxRealloc(kuM, (TuM+1)*sizeof(uint32_t)); 
	  if (kuM_expanded) { kuM = kuM_expanded; }
	  TuM += 1; //actually adding an extra table
	}
	new_t = new_table_idx; //== TuM if no previous 0 entry in nuM; otherwise, it equals the index of the 0 entry
	tuM[ee_i] = new_t + 1; 
	nuM[new_t] = 0; 
	int new_k = sampleDishFull(examps, resids, mC, muC, muD, kU, nC, nD, betaM, c0, sigmaSqd, invsigmaSqd, invsigmaSqd0, length_mC, numExamples_uu, rng);       
	kuM[new_t] = new_k + 1; //adding +1 because sampleDishFull returns idx from 0 to N-1, and we want to maintain matlab idxing in the arrays;
	//add table as needed
	if (new_k + 1 > length_mC){ //length_mC tells us number of active dishes right now
	  double* mC_expanded = (double*)mxRealloc(mC, (length_mC+1)*sizeof(double)); 
	  if (mC_expanded) { mC = mC_expanded; }
	  length_mC++; 
	  mC[new_k] = 0; //new dish
	} 
	mC[new_k] += 1; //updating the table to dishes count
	last_example_got_new_table = true;
      }
      else { 
	last_example_got_new_table = false; 	
      }
      nuM[new_t] += 1; //add 1 to ratings per table count
      tM[ee] = new_t + 1; //update the array over ALL examples to reflect the new table for a particular rating
      int new_k = kuM[new_t] - 1; //update dish sufficient stats
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
      kM[ee] = new_k + 1; 
      muC[new_k] = updateCRFMu(muC, nC, residC, new_k, true, model); //add sufficient statistics to dish parameters
      //NOTE: this is a hack!
      if (fabs(muC[new_k]) > 30){ int sign = (muC[new_k] > 0) - (muC[new_k] < 0); muC[new_k] = sign*30; }
      //if (fabs(muC[new_k]) < 0.0000000001){ int sign = (muC[new_k] > 0) - (muC[new_k] < 0); muC[new_k] = sign*0.0000000001; }
      nC[new_k] += 1;           
    } //close loop through examples per user         
    //update kuM, nuM (tuM remains same size, values are updated already, so no need to write out to plhs)
    mxArray* kuM_out; //for output; specific to each user
    mxArray* nuM_out; 
    mxArray* tuM_out; 
    kuM_out = mxCreateNumericMatrix(1, TuM, mxUINT32_CLASS, mxREAL); 
    if (kuM_out){
      mxSetPr(kuM_out, kuM); 
      mxSetCell(sampkuM_out, uu, kuM_out); 
    }
    else {mexPrintf("Problem with memory allocation for user %d\n", uu); exit(EXIT_FAILURE); }
    nuM_out = mxCreateNumericMatrix(1, TuM, mxUINT32_CLASS, mxREAL); 
    if (nuM_out){
      mxSetPr(nuM_out, nuM); 
      mxSetCell(sampnuM_out, uu, nuM_out);
    }
    else {mexPrintf("Problem with memory allocation for user %d\n", uu); exit(EXIT_FAILURE); }    
    tuM_out = mxCreateNumericMatrix(1, tuM_size, mxUINT32_CLASS, mxREAL); 
    if (tuM_out){
      mxSetPr(tuM_out, tuM); 
      mxSetCell(samptuM_out, uu, tuM_out); 
    }
    else {mexPrintf("Problem with memory allocation for user %d\n", uu); exit(EXIT_FAILURE); }
  } //close loop for looping through all users 
  //set up outputs for remaining variables, the ones that get modified across loops
  mxArray* mC_out = mxCreateDoubleMatrix(1, length_mC, mxREAL); 
  mxArray* nC_out = mxCreateDoubleMatrix(1, length_nC, mxREAL); 
  mxArray* muC_out = mxCreateDoubleMatrix(1, length_muC, mxREAL); 
  mxArray* tM_out = mxCreateNumericMatrix(1, length_tM, mxUINT32_CLASS, mxREAL); 
  mxArray* kM_out = mxCreateNumericMatrix(1, length_kM, mxUINT32_CLASS, mxREAL); 
  mxSetPr(mC_out, mC); 
  mxSetPr(nC_out, nC);
  mxSetPr(muC_out, muC);
  mxSetPr(tM_out, tM); //tM and kM do not change, so we can set their lengths 
  mxSetPr(kM_out, kM); 
  plhs[0] = sampkuM_out;
  plhs[1] = sampnuM_out; 
  plhs[2] = samptuM_out; 
  plhs[3] = mC_out;
  plhs[4] = nC_out; 
  plhs[5] = muC_out; 
  plhs[6] = tM_out; 
  plhs[7] = kM_out; 
  mexPrintf("Finished sampling tables\n"); 
}
