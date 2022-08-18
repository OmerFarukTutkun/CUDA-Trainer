#include "adam.h"
void initAdam(Adam *optimizer, NN *model, float lr, int clip)
{
    optimizer->model = model;
    optimizer->lr = lr;
    optimizer->clip = clip;
    optimizer->moments = (Moments *)malloc(sizeof(Moments) * model->num_of_layers);

    for (int i = 0; i < optimizer->model->num_of_layers; i++)
    {
        optimizer->moments[i].moment1_W = createMatrix(model->layers[i].weights->rows, model->layers[i].weights->columns);
        optimizer->moments[i].moment1_b = createMatrix(model->layers[i].biases->rows, model->layers[i].biases->columns);
        optimizer->moments[i].moment2_W = createMatrix(model->layers[i].weights->rows, model->layers[i].weights->columns);
        optimizer->moments[i].moment2_b = createMatrix(model->layers[i].biases->rows, model->layers[i].biases->columns);
    }
}
void AdamOptimizer(Adam *optimizer)
{
    for (int i = 0; i < optimizer->model->num_of_layers; i++)
    {
        int size = optimizer->moments[i].moment1_W->rows * optimizer->moments[i].moment1_W->columns;
        int nBlocks = (size - 1) / BlockSize + 1;
        AdamOptimizerKernel<<<nBlocks, BlockSize>>>(optimizer->model->layers[i].weights->data, optimizer->model->layers[i].weight_gradients->data,
                                                   optimizer->moments[i].moment1_W->data, optimizer->moments[i].moment2_W->data, size, optimizer->lr, beta1, beta2, epsilon, optimizer->clip);
        cudaDeviceSynchronize();
        size = optimizer->moments[i].moment1_b->rows * optimizer->moments[i].moment1_b->columns;
        nBlocks = (size - 1) / BlockSize + 1;
        AdamOptimizerKernel<<<nBlocks, BlockSize>>>(optimizer->model->layers[i].biases->data, optimizer->model->layers[i].bias_gradients->data,
                                                   optimizer->moments[i].moment1_b->data, optimizer->moments[i].moment2_b->data, size, optimizer->lr, beta1, beta2, epsilon, optimizer->clip);
        cudaDeviceSynchronize();
    }
}
void freeAdam(Adam *optimizer)
{
    for (int i = 0; i < optimizer->model->num_of_layers; i++)
    {
        freeMatrix(optimizer->moments[i].moment1_W);
        freeMatrix(optimizer->moments[i].moment1_b);
        freeMatrix(optimizer->moments[i].moment2_W);
        freeMatrix(optimizer->moments[i].moment2_b);
    }
    free(optimizer->moments);
}