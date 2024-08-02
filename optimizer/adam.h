
#ifndef _ADAM_H_
#define _ADAM_H_

#include "../network/network.h"

typedef struct Moments
{
    Matrix *moment1_W;
    Matrix *moment1_b;
    Matrix *moment2_W;
    Matrix *moment2_b;
} Moments;

typedef struct SlowParams
{
    Matrix *weights;
    Matrix *biases;
}SlowParams;

typedef struct Adam
{
    int k;
    float alpha;
    int step_size;
    int clip;
    float lr;
    Moments *moments;
    SlowParams * slowParams;
    NN *model;
} Adam;

static const float beta1 = 0.9;
static const float beta2 = 0.999;
static const float epsilon = 1e-8;

static const float clip_min = -1.98;
static const float clip_max = 1.98;

__global__ void AdamOptimizerKernel(float *values, float *gradients, float *first_moment, float *second_moment, int size, float alpha, float beta1, float beta2, float eps, int clip);
void initAdam(Adam *optimizer, NN *model, float lr, int clip);
void AdamOptimizer(Adam *optimizer);
void freeAdam(Adam *opt);
#endif