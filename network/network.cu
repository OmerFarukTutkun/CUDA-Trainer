#include "network.h"
void initNN(NN *model)
{
    model->num_of_layers = 2;
    model->layers = (Layer *)malloc(sizeof(Layer) * model->num_of_layers);
 
    initLayer(FeatureTransformer, INPUT_SIZE, L1, model->layers, (Activation){.apply = relu , .backprop= backpropRelu}, BATCH_SIZE);
    initLayer(Linear, 2 *L1, 1, model->layers + 1, (Activation){.apply = sigmoid , .backprop= backpropSigmoid}, BATCH_SIZE);


    model->loss = (Loss){.apply = MSE, .backprop = backpropMSE};
}

void forward_model(int32_t *feature_indices_us, int32_t *feature_indices_enemy, NN *model)
{
    forwardFeatureTransformer(feature_indices_us, feature_indices_enemy, model->layers);
    for (int i = 1; i < model->num_of_layers; i++)
        forwardLinear(model->layers[i - 1].output, model->layers + i);
}
void backward_model(NN *model, Matrix *target, Matrix *loss, int32_t *feature_indices_us, int32_t *feature_indices_enemy)
{
    model->loss.apply(model->layers[model->num_of_layers - 1].output.activated, target, loss);
    model->loss.backprop(model->layers[model->num_of_layers - 1].output.activated, target, model->layers[model->num_of_layers - 1].output.gradients);

    for (int i = model->num_of_layers - 1; i > 0; i--)
        backwardLinear(model->layers[i - 1].output, model->layers + i);
    backwardFeatureTransformer(feature_indices_us, feature_indices_enemy, &model->layers[0]);
}
void saveNN(NN *model, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    for(int i=0; i<model->num_of_layers ; i++)
    {
        fwrite(model->layers[i].weights->data , sizeof(float) ,model->layers[i].weights->rows* model->layers[i].weights->columns , file);
        fwrite(model->layers[i].biases->data , sizeof(float) ,model->layers[i].biases->rows* model->layers[i].biases->columns , file);
    }
    fclose(file);
}
void readNN(NN *model, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    for(int i=0; i<model->num_of_layers ; i++)
    {
        fread(model->layers[i].weights->data , sizeof(float) ,model->layers[i].weights->rows* model->layers[i].weights->columns , file);
        fread(model->layers[i].biases->data , sizeof(float) ,model->layers[i].biases->rows* model->layers[i].biases->columns , file);
    }
    fclose(file);
}
void freeNN(NN *model)
{
    for (int i = 0; i < model->num_of_layers; i++)
    {
        freeLayer(model->layers + i);
    }
    free(model->layers);
}