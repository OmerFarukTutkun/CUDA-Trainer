#include "adam.h"
__global__ void AdamOptimizerKernel(float *values, float *gradients, float *first_moment, float *second_moment, int N, float alpha, float beta1, float beta2, float eps, int clip)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        first_moment[index] = beta1 * first_moment[index] + (1 - beta1) * gradients[index];
        second_moment[index] = beta2 * second_moment[index] + (1 - beta2) * gradients[index] * gradients[index];

        float delta = alpha * first_moment[index] / (sqrtf(second_moment[index]) + eps);

        values[index] *= (1.0 - alpha*0.01);

        values[index] -= delta;
        if(clip)
            values[index] = MIN(clip_max , MAX(clip_min , values[index]));
        gradients[index] = 0;
    }
}