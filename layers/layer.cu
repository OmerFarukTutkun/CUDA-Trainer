#include "layer.h"
void forwardFeatureTransformer(int32_t *feature_indices_us, int32_t *feature_indices_enemy, Layer *layer)
{
    feature_transformer_slice_forward<<<BATCH_SIZE, BlockSize>>>(feature_indices_us, layer->weights->data, layer->biases->data, layer->output.unactivated->data);
    cudaDeviceSynchronize();
    feature_transformer_slice_forward<<<BATCH_SIZE, BlockSize>>>(feature_indices_enemy, layer->weights->data, layer->biases->data, &layer->output.unactivated->data[L1]);
    cudaDeviceSynchronize();
    layer->activation.apply(layer->output.unactivated, layer->output.activated);
}
void backwardFeatureTransformer(int32_t *feature_indices_us, int32_t *feature_indices_enemy, Layer *layer)
{
    layer->activation.backprop(layer->output.unactivated, layer->output.activated, layer->output.gradients);
    feature_transformer_slice_backward<<<BATCH_SIZE, BlockSize>>>(feature_indices_us, layer->weight_gradients->data, layer->bias_gradients->data, layer->output.gradients->data);
    cudaDeviceSynchronize();
    feature_transformer_slice_backward<<<BATCH_SIZE, BlockSize>>>(feature_indices_enemy, layer->weight_gradients->data, layer->bias_gradients->data, &layer->output.gradients->data[L1]);
    cudaDeviceSynchronize();
}
void forwardLinear(Output previousLayerOutput, Layer *layer)
{
    MultipyMatrix(previousLayerOutput.activated, layer->weights, layer->output.unactivated, 0, 1);
    addVectorToMatrix(layer->output.unactivated, layer->biases);
    layer->activation.apply(layer->output.unactivated, layer->output.activated);
}
void backwardLinear(Output previousLayerOutput, Layer *layer)
{
    layer->activation.backprop(layer->output.unactivated, layer->output.activated, layer->output.gradients);
    sumRowsOfMatrix(layer->output.gradients, layer->bias_gradients);
    MultipyMatrix(layer->output.gradients, layer->weights, previousLayerOutput.gradients, 0, 0);
    MultipyMatrix(layer->output.gradients, previousLayerOutput.activated, layer->weight_gradients, 1, 0);
}
void initLayer(int layerType, int inputSize, int outputSize, Layer *layer, Activation act, int batchSize)
{
    layer->activation = act;
    if (layerType == FeatureTransformer)
    {
        layer->weights = createMatrix(inputSize, outputSize);
        randomizeMatrix(layer->weights, sqrt(2.0f / inputSize), 0.0f);
        layer->biases = createMatrix(outputSize, 1);
        layer->bias_gradients = createMatrix(outputSize, 1);
        layer->weight_gradients = createMatrix(inputSize, outputSize);

        layer->output.activated = createMatrix(batchSize, 2 * outputSize);
        layer->output.unactivated = createMatrix(batchSize, 2 * outputSize);
        layer->output.gradients = createMatrix(batchSize, 2 * outputSize);
    }
    else if (layerType == Linear)
    {
        layer->weights = createMatrix(outputSize, inputSize);
        randomizeMatrix(layer->weights, sqrt(2.0f / inputSize), 0.0f);
        layer->biases = createMatrix(outputSize, 1);
        layer->bias_gradients = createMatrix(outputSize, 1);
        layer->weight_gradients = createMatrix(outputSize, inputSize);

        layer->output.activated = createMatrix(batchSize, outputSize);
        layer->output.unactivated = createMatrix(batchSize, outputSize);
        layer->output.gradients = createMatrix(batchSize, outputSize);
    }
}
void freeLayer(Layer *layer)
{
    freeMatrix(layer->weight_gradients);
    freeMatrix(layer->weights);
    freeMatrix(layer->bias_gradients);
    freeMatrix(layer->biases);
    freeMatrix(layer->output.activated);
    freeMatrix(layer->output.unactivated);
    freeMatrix(layer->output.gradients);
}