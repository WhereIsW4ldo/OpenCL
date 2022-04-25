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



__kernel void convolution_3_x_3(
    int size, __global float *matrix, 
    __global float *_kernel, __global float *out)
{
    int i, j;
	float sum;
	float zeropad[SIZE + 2][SIZE + 2] = { {0.} };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			//zeropad[i + 1][j + 1] = matrix[i][j];
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