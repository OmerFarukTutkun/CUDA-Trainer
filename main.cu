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
	initAdam(optimizer, model, LR, true);

	FILE *file = fopen(TRANINNG_FILE, "rb");
	FILE *log = fopen("training.log", "w");

	time_t tm;
	time(&tm);
	fprintf(log, "Date = %s", ctime(&tm));
	fprintf(log, "Arch name: %s\n", ARCH_NAME);

	Data *buffer = (Data *)malloc(sizeof(Data) * BATCH_SIZE);

	int32_t *feature_indices_us_cpu = (int32_t *)malloc(MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	int32_t *feature_indices_enemy_cpu = (int32_t *)malloc(MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	int32_t *feature_indices_us_gpu;
	int32_t *feature_indices_enemy_gpu;

	cudaMalloc(&feature_indices_us_gpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);
	cudaMalloc(&feature_indices_enemy_gpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE);

	float *eval_cpu = (float *)malloc(sizeof(float) * BATCH_SIZE);
	Matrix *eval_gpu = createMatrix(BATCH_SIZE, 1);

	float *result_cpu = (float *)malloc(sizeof(float) * BATCH_SIZE);
	Matrix *result_gpu = createMatrix(BATCH_SIZE, 1);

	Matrix *target = createMatrix(BATCH_SIZE, 1);
	
	if (file == NULL)
	{
		printf("Traning data could not open\n");
		return 0;
	}
	printf("training started\n");
	int number_of_epoch = 50;
	long int clk = clock();
	for (int i = 0; i < 6000 * number_of_epoch; i++)
	{
		if (!fread(buffer, sizeof(Data), BATCH_SIZE, file))
			break;
		load_data(buffer, BATCH_SIZE, feature_indices_us_cpu, feature_indices_enemy_cpu, eval_cpu, result_cpu);

		cudaMemcpy(feature_indices_enemy_gpu, feature_indices_enemy_cpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(feature_indices_us_gpu, feature_indices_us_cpu, MAX_ACTIVE_FEATURE * sizeof(int32_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(eval_gpu->data, eval_cpu, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(result_gpu->data, result_cpu, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		linearFunction(eval_gpu, eval_gpu, SigmoidCoefficient, 0.0f);
		sigmoid(eval_gpu, eval_gpu);

		linearFunction3Variable(eval_gpu, result_gpu, target, LAMBDA, 1 - LAMBDA, 0.0f);
		forward_model(feature_indices_us_gpu, feature_indices_enemy_gpu, model);
		backward_model(model, target, loss, feature_indices_us_gpu, feature_indices_enemy_gpu);
		AdamOptimizer(optimizer);
		

		if (i % 1000 == 999)
		{

			float average_loss = sumMatrix(loss) / (1000 * BATCH_SIZE);
			float time_in_sec = (clock() - clk) / (float)CLOCKS_PER_SEC;
			int nps = (1000 * BATCH_SIZE) / time_in_sec;
			printf("step: %10d  loss : %10f 	nps: %10d\n", i + 1, average_loss, nps);
			fprintf(log, "step: %10d  loss : %10f 	nps: %10d\n", i + 1, average_loss, nps);
			fclose(log);
			log = fopen("training.log", "a");
			zeroMatrix(loss);
			clk = clock();
		}
		if (i % 6000 == 5999)
		{
			char filename[100];
			sprintf(filename, "devre_epoch%d.nnue", (i + 1) / 6000);
			saveNN(model, filename);
			printf("%d. epoch finished\n", i / 6000 + 1);
		}
	}

	saveNN(model, "devre.nnue");

	freeMatrix(loss);
	freeAdam(optimizer);
	freeNN(model);

	fclose(file);
	fclose(log);
	free(buffer);
	free(feature_indices_enemy_cpu);
	free(feature_indices_us_cpu);
	cudaFree(feature_indices_enemy_gpu);
	cudaFree(feature_indices_us_gpu);

	free(result_cpu);
	freeMatrix(result_gpu);
	free(eval_cpu);
	freeMatrix(eval_gpu);
	freeMatrix(target);

	printf("training finished\n");
	return 0;
}