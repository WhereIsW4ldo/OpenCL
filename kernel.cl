#define SIZE 224
#define NUM_CHANNELS 3
#define CONV_SIZE 3
#define KERNEL_SIZE 3
#define NUM_LAYERS 13
#define NUM_DENSE 3


#define IMAGE_INDEX(l, x, y)\
	((l) * (SIZE) * (SIZE) + (y) * (SIZE) + (x))	
#define PLANE_INDEX(l)\
    ((l) * (SIZE) * (SIZE))
#define MEM_BLOCK_DEPTH 512


void conv_3x3 (int size, float *matrix, float *_kernel, float *out)
{
	int i, j;
	float sum;
	float zeropad[SIZE + 2][SIZE + 2] = { {0.} };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[(size * i) + j];
		}
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropad[i][j] * _kernel[0 * 3 + 0] +
				zeropad[i + 1][j] * _kernel[1 * 3 + 0] +
				zeropad[i + 2][j] * _kernel[2 * 3 + 0]+
				zeropad[i][j + 1] * _kernel[0 * 3 + 1]+
				zeropad[i + 1][j + 1] * _kernel[1 * 3 + 1]+
				zeropad[i + 2][j + 1] * _kernel[2 * 3 + 1]+
				zeropad[i][j + 2] * _kernel[0 * 3 + 2]+
				zeropad[i + 1][j + 2] * _kernel[1 * 3 + 2]+
				zeropad[i + 2][j + 2] * _kernel[2 * 3 + 2];
			// out[i][j] += sum;
			out[(size * i) + j] += sum;
		}
	}
}

void add_bias_and_relu(int size, float *out, float bs) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[(i * size) + j] += bs;
			if (out[(i * size) + j] < 0)
				out[(i * size) + j] = 0.0;
		}
	}
}

__kernel void gpu_shenanigans(
    int feature_size, int input_depth, int output_depth, __global float *input_features, 
    __global float *layer_weights, __global float *layer_biases, __global float *output_features)
{

	float* input  = &input_features;
	float* output = &output_features;
	float* weight = &layer_weights;

	for (int output_it = 0; output_it < output_depth; output_it++) 
	{
		for (int input_it = 0; input_it < input_depth; input_it++) 
		{
			conv_3x3(feature_size, &input[input_it * feature_size * feature_size],
							  &weight[output_it * input_depth * CONV_SIZE * CONV_SIZE +
							  				 input_it * CONV_SIZE * CONV_SIZE],
							  &output[output_it * feature_size * feature_size]);
		}
		add_bias_and_relu(feature_size, &output[output_it * feature_size * feature_size], layer_biases[output_it]);
	}
}