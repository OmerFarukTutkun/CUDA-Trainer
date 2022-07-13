#include "operations.h"

__global__ void linearFunction3VariableKernel(float *A, float *B, float *C, float alpha, float beta, float theta, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		C[index] = alpha * A[index] + beta * B[index] + theta;
}
__global__ void linearFunctionKernel(float *A, float *B, float alpha, float beta, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		B[index] = alpha * A[index] + beta;
}
__global__ void clipMatrixKernel(float *A, float min, float max, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		A[index] = MAX(min, MIN(max, A[index]));
}

// https://github.com/glinscott/nnue-pytorch/
__global__ void feature_transformer_slice_forward(int32_t *feature_indices, float *weight, float *bias, float *output)
{

	const uint32_t max_active_features = MAX_ACTIVE_FEATURE;
	const uint32_t output_thread_slice_size = L1 / BlockSize;
	const uint32_t output_size = L1;

	__shared__ float shared_output[output_size];

	uint32_t block_idx = blockIdx.x;
	uint32_t slice_offset = threadIdx.x * output_thread_slice_size;

	float *output_slice = output + block_idx * 2 * output_size + slice_offset;
	float *bias_slice = bias + slice_offset;
	float *shared_output_slice = shared_output + slice_offset;
	int32_t *feature_index_row = feature_indices + block_idx * max_active_features;
	for (uint32_t s = 0; s < output_thread_slice_size; ++s)
	{
		shared_output_slice[s] = bias_slice[s];
	}

	for (uint32_t k = 0; k < max_active_features; ++k)
	{
		int32_t feature_index = feature_index_row[k];
		if (feature_index != -1)
		{
			float *weight_slice = weight + feature_index * output_size + slice_offset;
			for (uint32_t s = 0; s < output_thread_slice_size; ++s)
			{
				shared_output_slice[s] += weight_slice[s];
			}
		}
		else
			break;
	}
	for (uint32_t s = 0; s < output_thread_slice_size; ++s)
	{
		output_slice[s] = shared_output_slice[s];
	}
}
__global__ void feature_transformer_slice_backward(int32_t *feature_indices, float *weight_grad, float *bias_grad, float *output_grad)
{
	const uint32_t max_active_features = MAX_ACTIVE_FEATURE;
	const uint32_t output_thread_slice_size = L1 / BlockSize;
	const uint32_t output_size = L1;

	__shared__ float shared_output_grad[output_size];
	int block_idx = blockIdx.x;
	int slice_offset = threadIdx.x * output_thread_slice_size;

	float *output_grad_slice = output_grad + block_idx * 2 * output_size + slice_offset;
	float *bias_grad_slice = bias_grad + slice_offset;
	float *shared_output_grad_slice = shared_output_grad + slice_offset;
	int32_t *feature_index_row = feature_indices + block_idx * max_active_features;

	for (uint32_t s = 0; s < output_thread_slice_size; ++s)
	{
		shared_output_grad_slice[s] = output_grad_slice[s];
	}

	for (uint32_t s = 0; s < output_thread_slice_size; ++s)
	{
		const float sog = shared_output_grad_slice[s];
		if (sog != 0.0f)
		{
			atomicAdd(&bias_grad_slice[s], sog);
			for (uint32_t k = 0; k < max_active_features; ++k)
			{
				int32_t feature_index = feature_index_row[k];
				if (feature_index != -1)
				{
					float *weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
					atomicAdd(&weight_grad_slice[s], sog);
				}
				else
					break;
			}
		}
	}
}
__global__ void addVectorToMatrixKernel(float *A, float *v, int m, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		A[index] = A[index] + v[index % m];
}