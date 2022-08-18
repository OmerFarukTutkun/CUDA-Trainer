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

#define TRANINNG_FILE ""

#define SigmoidCoefficient (1.0f / 250.0f)
#define BlockSize 256

#define BATCH_SIZE 16384
#define MAX_ACTIVE_FEATURE 32
#define LR 0.002

#define LAMBDA 1.0 // 1.0: purely on eval , 0.0: purely on game result
#define INPUT_SIZE 32*768
#define L1 512

#define FT_BLOCK_SIZE MIN(L1, BlockSize)
#endif
