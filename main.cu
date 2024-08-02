#include "activations/activations.h"
#include "loss/loss.h"
#include "operations/matrix.h"
#include "operations/operations.h"
#include "network/network.h"
#include "optimizer/adam.h"
#include "training_data_loader.h"
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>


int main(int argc, char **argv )
{

    time_t curr_time;
    tm * curr_tm;
    char tempChArray[100];
    time(&curr_time);
    curr_tm = localtime(&curr_time);
    strftime(tempChArray, 50, "%m.%d.%Y_%T", curr_tm);

    int result = mkdir(tempChArray, 0777);
    if(result != 0)
    {
        std::cout << "Cannot mkdir: " << tempChArray << std::endl;
        return 0;
    }

    std::string folderName(tempChArray);

	srand(time(NULL));
	initOnes();
	NN *model = (NN *)malloc(sizeof(NN));

	Matrix *loss = createMatrix(BATCH_SIZE, 1);

	Adam *optimizer = (Adam *)malloc(sizeof(Adam));

	initNN(model);
	initAdam(optimizer, model, LR, true);

	FILE *file = fopen(TRANINNG_FILE, "rb");
	FILE *logFile = fopen((folderName + "/training.log").c_str(), "w");

	time_t tm;
	time(&tm);
	fprintf(logFile, "Date = %s", ctime(&tm));
	fprintf(logFile, "Arch name: %s\n", ARCH_NAME);

    constexpr int iterPerEpoch = 100000000/BATCH_SIZE;
    constexpr auto bufferMemory = sizeof(Board) * BATCH_SIZE*2*iterPerEpoch; //large memory(6.4gb) to 200m read data at once and shuffle it

    auto *buffer = (Board *)malloc(bufferMemory);
    constexpr int numBatches =  bufferMemory / (sizeof(Board) * BATCH_SIZE);


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
	int number_of_epoch = 500;
	long int clk = clock();
    double dLoss = 0.0f;

	for (int i = 0; i < iterPerEpoch * number_of_epoch; i++)
	{
        int index = i % numBatches;
        if(index == 0) {
            if (numBatches * BATCH_SIZE != fread(buffer, sizeof(Board), numBatches * BATCH_SIZE, file)) {
                //return back to the start of the file
                fseek(file, 0, SEEK_SET);

                if (numBatches * BATCH_SIZE != fread(buffer, sizeof(Board), numBatches * BATCH_SIZE, file)) {
                    break;
                }
            }
            std::shuffle(buffer, buffer + numBatches * BATCH_SIZE, std::mt19937(std::random_device()()));
        }

		load_data(buffer + index*BATCH_SIZE, BATCH_SIZE, feature_indices_us_cpu, feature_indices_enemy_cpu, eval_cpu, result_cpu);

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
		
        dLoss += sumMatrix(loss)/BATCH_SIZE;
        zeroMatrix(loss);

		if (i % iterPerEpoch == iterPerEpoch -1)
		{
			printf("%d. epoch finished\n", i / iterPerEpoch + 1);

            if((i + 1) / iterPerEpoch % 100 == 0)
                optimizer->lr = 0.3*optimizer->lr;

            if((i + 1) / iterPerEpoch % 10 == 0)
            {
                char filename[100];
                sprintf(filename, (folderName + "/devre_epoch%d.nnue").c_str(), (i + 1) / iterPerEpoch);
                saveNN(model, filename);
            }

            float average_loss = dLoss / iterPerEpoch;
            float time_in_sec = (clock() - clk) / (float)CLOCKS_PER_SEC;
            int nps = (iterPerEpoch * BATCH_SIZE) / time_in_sec;

            printf("epoch: %10d  loss : %10f 	nps: %10d\n", (i + 1) / iterPerEpoch, average_loss, nps);
            fprintf(logFile, "epoch: %10d  loss : %10f 	nps: %10d\n", (i + 1) / iterPerEpoch, average_loss, nps);
            fclose(logFile);
            logFile = fopen((folderName + "/training.log").c_str(), "a");

            clk = clock();
            dLoss = 0.0f;
		}
	}

	saveNN(model, (folderName + "/devre.nnue").c_str());

	freeMatrix(loss);
	freeAdam(optimizer);
	freeNN(model);

	fclose(file);
	fclose(logFile);
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