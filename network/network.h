#ifndef _NN_H_
#define _NN_H_
#include "../activations/activations.h"
#include "../layers/layer.h"
#include "../loss/loss.h"

#define ARCH_NAME "32*768->(2x512)->1"

typedef struct NN
{
    Layer *layers;
    Loss loss;
    int num_of_layers;
} NN;
void initNN(NN *model);
void forward_model(int32_t *feature_indices_us, int32_t *feature_indices_enemy, NN *model);
void backward_model(NN *model, Matrix *target, Matrix *loss, int32_t *feature_indices_us, int32_t *feature_indices_enemy);
void saveNN(NN *model, const char *filename);
void readNN(NN *model, const char *filename);
void freeNN(NN *model);
#endif