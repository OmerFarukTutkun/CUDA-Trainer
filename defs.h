#ifndef _DEFS_H_
#define _DEFS_H_

#include <assert.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define TRANINNG_FILE "trainingData"


#define SigmoidCoefficient (1.0f / 400.0f)

#define FT_BLOCK_SIZE 768
#define BlockSize 512 //1024 is the maximum we can make
#define BATCH_SIZE 16384


#define MAX_ACTIVE_FEATURE 32
static double LR = 0.001;

#define LAMBDA 1.0 // 1.0: purely on eval , 0.0: purely on game result
#define INPUT_SIZE 768
#define L1 1536


#endif
