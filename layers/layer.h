#ifndef _LAYER_H_
#define _LAYER_H_
#include "../activations/activations.h"
#include "../operations/matrix.h"
#include "../operations/operations.h"

typedef struct Output
{
    Matrix *unactivated;
    Matrix *activated;
    Matrix *gradients;
} Output;

enum LayerTypes
{
    FeatureTransformer,
    Linear,
};
typedef struct Layer
{
    Matrix *weights;
    Matrix *biases;
    Matrix *weight_gradients;
    Matrix *bias_gradients;
    Activation activation;
    Output output;
} Layer;

void forwardFeatureTransformer(int32_t *feature_indices_us, int32_t *feature_indices_enemy, Layer *layer);
void backwardFeatureTransformer(int32_t *feature_indices_us, int32_t *feature_indices_enemy, Layer *layer);

void forwardLinear(Output previousLayerOutput, Layer *layer);
void backwardLinear(Output previousLayerOutput, Layer *layer);

void initLayer(int layerType, int inputSize, int outputSize, Layer *layer, Activation act, int batchSize);
void freeLayer(Layer *layer);
#endif