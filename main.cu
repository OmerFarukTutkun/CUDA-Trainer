#include "activations/activations.h"
#include "loss/loss.h"
#include "operations/matrix.h"
#include "operations/operations.h"
#include "network/network.h"
#include "optimizer/adam.h"
#include "training_data_loader.h"
#include <unistd.h>

int main()
{
	srand(time(NULL));
	initOnes();
	NN *model = (NN *)malloc(sizeof(NN));

	Matrix *loss = createMatrix(BATCH_SIZE, 1);

	Adam *optimizer = (Adam *)malloc(sizeof(Adam));

	initNN(model);
	initAdam(optimizer, model, LR , true);

	FILE *file = fopen(TRANINNG_FILE, "rb");
	Data *buffer = (Data *)malloc(sizeof(Data) * BATCH_SIZE);

	int32_t *feature_indices_us_cpu = (int32_t *)malloc(MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	int32_t *feature_indices_enemy_cpu = (int32_t *)malloc(MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	int32_t *feature_indices_us_gpu;
	int32_t *feature_indices_enemy_gpu;

	cudaMalloc(&feature_indices_us_gpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	cudaMalloc(&feature_indices_enemy_gpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);

	float *target_cpu = (float *)malloc(sizeof(float) * BATCH_SIZE);
	Matrix *target_gpu = createMatrix(BATCH_SIZE, 1);

	if (file == NULL)
	{
		printf("Traning data could not open\n");
		return 0;
	}
	printf("training started\n");
	int number_of_epoch = 50;
	for (int i = 0; i < 6000 * number_of_epoch; i++)
	{
		long int start_clk;
		if (i % 1000 == 0)
		{
			start_clk = clock();
		}
		if (!fread(buffer, sizeof(Data), BATCH_SIZE, file))
			break;
		load_data(buffer, BATCH_SIZE, feature_indices_us_cpu, feature_indices_enemy_cpu, target_cpu);

		cudaMemcpy(feature_indices_enemy_gpu, feature_indices_enemy_cpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(feature_indices_us_gpu, feature_indices_us_cpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(target_gpu->data, target_cpu, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		linearFunction(target_gpu, target_gpu, SigmoidCoefficient, 0.0f);
		sigmoid(target_gpu, target_gpu);

		forward_model(feature_indices_us_gpu, feature_indices_enemy_gpu, model);
		backward_model(model, target_gpu, loss, feature_indices_us_gpu, feature_indices_enemy_gpu);
		AdamOptimizer(optimizer);
		if (i % 1000 == 999)
		{
			printf("step: %d  ", i + 1);
			printf("loss : %f", sumMatrix(loss) / (1000 * BATCH_SIZE));
			float time_in_sec = (clock() - start_clk) / (float)CLOCKS_PER_SEC;
			float nps = (1000 * BATCH_SIZE) / time_in_sec;
			printf("	nps: %ld\n", (long int)nps);
			zeroMatrix(loss);
		}
		if (i % 6000 == 5999)
		{
			char filename[100];
			sprintf(filename, "devre_epoch%d.nnue", (i + 1)/ 6000);
			saveNN(model, filename);
			printf("%d. epoch finished\n", i / 6000 + 1);
		}
	}

	freeMatrix(loss);
	freeAdam(optimizer);
	freeNN(model);

	fclose(file);
	free(buffer);
	free(feature_indices_enemy_cpu);
	free(feature_indices_us_cpu);
	cudaFree(feature_indices_enemy_gpu);
	cudaFree(feature_indices_us_gpu);

	free(target_cpu);
	freeMatrix(target_gpu);

	printf("training finished\n");
	return 0;
}