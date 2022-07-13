#ifndef _DEFS_H_
#define _DEFS_H_

#include <assert.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define TRANINNG_FILE "farseer_shuffled.bin"

#define SigmoidCoefficient (1.0f / 280.0f)
#define BlockSize 256

#define BATCH_SIZE 16384 // needs to be power of 32
#define MAX_ACTIVE_FEATURE 32
#define LR 0.002

#define INPUT_SIZE 768
#define L1 512

#endif