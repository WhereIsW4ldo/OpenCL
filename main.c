/* 
	Pretrained VGG16 convolutional neural network in C language
	GitHUB Page: https://github.com/ZFTurbo/VGG16-Pretrained-C
	Author: ZFTurbo
	
	Compilation: gcc -O3 -fopenmp -lm ZFC_VGG16_CPU.c -o ZFC_VGG16_CPU.exe
	Usage: ZFC_VGG16_CPU.exe <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: ZFC_VGG16_CPU.exe "weights.txt" "image_list.txt" "results.txt" 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

//#include <opencv2/opencv.hpp>

#include "./common/time_utils.h"
#include "./common/ocl_utils.h"
#include "imagenet_labels.h"

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


// Weights and image block START
float *image;
int cshape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};
float *wc[NUM_LAYERS];
float *bc[NUM_LAYERS];
int dshape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};
float *wd[NUM_DENSE];
float *bd[NUM_DENSE];

cl_mem matrix;
cl_mem kernel_;
cl_mem output;
cl_kernel kernel;
float *host_result;

// Blocks for intermediate convolutions
int mem_block_shape[3] = {MEM_BLOCK_DEPTH, SIZE, SIZE};

float *mem_block1;
float *mem_block2;
// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 7 * 7 };
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END


void reset_mem_block(float *mem) {
	memset(mem, 0, MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
}


void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}


void init_memory() {
	int l;

	// Init image memory
    image = (float *)malloc(NUM_CHANNELS * SIZE * SIZE * sizeof(float));

	// Init convolution weights
	for (l = 0; l < 13; l++) {
		wc[l] = malloc(cshape[l][0] * cshape[l][1] * cshape[l][2] * cshape[l][3] * sizeof(float));
		bc[l] = malloc(cshape[l][0] * sizeof(float));
	}

	// Init dense weights
	for (l = 0; l < 3; l++) {
		wd[l] = malloc(dshape[l][0] * dshape[l][1] * sizeof(float));
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks
	mem_block1 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
	mem_block2 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);
}


void free_memory() {
	int l;

	// Free image memory
    free(image);

	// Free convolution weights
	for (l = 0; l < 13; l++) {
		free(wc[l]);
		free(bc[l]);
	}

	// Free dense weights
	for (l = 0; l < 3; l++) {
		free(wd[l]);
		free(bd[l]);
	}

	// Free memblocks
	free(mem_block1);
	free(mem_block2);

}

cl_mem create_and_init_vector(void)
{
	int grootte = SIZE;
    cl_int error;
    cl_float *host_vec = (cl_float*) malloc(sizeof(cl_float) * grootte * grootte);
    for (int i = 0; i < grootte * grootte; ++i)
        host_vec[i] = i;
    cl_mem dev_vec = clCreateBuffer(g_context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * grootte * grootte, host_vec, &error);
    ocl_err(error);
    ocl_err(clFinish(g_command_queue));
    free(host_vec);
	printf("created dev vector\n");
    return dev_vec;
}

cl_mem create_result_buffer(void)
{
    cl_int error;
    float *host_vec = (float*) malloc(sizeof(cl_float) * SIZE * SIZE);
    for (int i = 0; i < SIZE * SIZE; ++i)
        host_vec[i] = 0;

    cl_mem dev_vec = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * SIZE * SIZE, host_vec, &error);
    ocl_err(error);
    return dev_vec;
}


/*
 * read_weights is te negeren
*/
void read_weights(char *in_file, int lvls) {
	float dval;
	int i, j, k, l, z;
	FILE *iin;
	int total_lvls_read = 0;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	
	// Reading convolution weights (store them flipped from begining)
	for (z = 0; z < 13; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read conv block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						wc[z][i * cshape[z][1] * cshape[z][2] * cshape[z][3] +
							  j * cshape[z][2] * cshape[z][3] +
							  (CONV_SIZE - k  - 1)* cshape[z][3] +
							  CONV_SIZE - l - 1] = dval;
					}
				}
			}
		}
		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			bc[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	// Reading dense weights
	for (z = 0; z < 3; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read dense block %d weights\n", z);
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				wd[z][i * dshape[z][1] + j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			bd[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	fclose(iin);
}


void normalize_image() {
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[IMAGE_INDEX(l, i, j)] -= coef[l];
			}
		}
	}
}

void store_image(const unsigned char *input)
{
	int i, j, l;

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[IMAGE_INDEX(l, i, j)] = input[j * SIZE * 3 + i * 3 + 2 - l]; 
			}
		}
	}
}

// function in kernel
void convolution_3_x_3(int size, float matrix[][size], float kernel[][CONV_SIZE],
                       float out[][size]) {
	int i, j;
	float sum;
	float zeropad[SIZE + 2][SIZE + 2] = { {0.} };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}	
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropad[i][j] * kernel[0][0] +
				zeropad[i + 1][j] * kernel[1][0] +
				zeropad[i + 2][j] * kernel[2][0] +
				zeropad[i][j + 1] * kernel[0][1] +
				zeropad[i + 1][j + 1] * kernel[1][1] +
				zeropad[i + 2][j + 1] * kernel[2][1] +
				zeropad[i][j + 2] * kernel[0][2] +
				zeropad[i + 1][j + 2] * kernel[1][2] +
				zeropad[i + 2][j + 2] * kernel[2][2];
			out[i][j] += sum;
		}
	}
}

void add_bias_and_relu(int size, float out[][size], float bs) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;
		}
	}
}

void convolution_layer(int feature_size, int input_depth, int output_depth,
					   float *input_features, float *layer_weights, float *layer_biases, float *output_features)
{
	for (int output_it = 0; output_it < output_depth; output_it++) {
		for (int input_it = 0; input_it < input_depth; input_it++) {
			clEnqueueWriteBuffer(g_command_queue, matrix, CL_TRUE, 0, sizeof(cl_float) * feature_size * feature_size, 
									&input_features[(output_it * input_depth * CONV_SIZE * CONV_SIZE) +
									(input_it * CONV_SIZE * CONV_SIZE)], 0, NULL, NULL);

			clEnqueueWriteBuffer(g_command_queue, kernel_, CL_TRUE, 0, sizeof(cl_float) * feature_size * feature_size, 
									&layer_weights[(output_it * input_depth * CONV_SIZE * CONV_SIZE) +
									(input_it * CONV_SIZE * CONV_SIZE)], 0, NULL, NULL);

			// Set kernel arguments
			int arg_num = 0;
			ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &feature_size));
			ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &matrix));
			ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &kernel_));
			ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &output));

			// printf("output_it = %d, input_depth = %d, feature_size = %d, input_it = %d\n", output_it, input_depth, feature_size, input_it);
			// printf("%d\n", ((output_it * input_depth * feature_size * feature_size) + (input_it * feature_size * feature_size)) <= (SIZE * SIZE * MEM_BLOCK_DEPTH));
			// Read result
			ocl_err(clEnqueueReadBuffer(g_command_queue, output, CL_TRUE,
					0, sizeof(cl_float) * SIZE*SIZE, &output_features[output_it * feature_size * feature_size + (input_it * feature_size)], 0, NULL, NULL));
		}
		
		add_bias_and_relu(feature_size, &output_features[output_it * feature_size * feature_size], layer_biases[output_it]);
	}
}



void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}


float max_of_4(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}


void maxpooling(int size, int depth, float *feature_map) {
	int i, j, d;

	float temp_output[depth * (size / 2) * (size / 2)];
	for (d = 0; d < depth; d++) {
		float *in = &feature_map[d * size * size];
		float *out = &temp_output[d * (size / 2) * (size / 2)];
		for (i = 0; i < size; i+=2) {
			for (j = 0; j < size; j+=2) {
				out[(i / 2) * (size / 2) + j / 2] = max_of_4(in[i * size + j],
													   in[(i + 1) * size + j],
													   in[i * size + j + 1],
													   in[(i + 1) * size + j + 1]);
			}
		}
	}
	memcpy(feature_map, temp_output, depth * (size / 2) * (size / 2) * sizeof(float));
}


void flatten(int sh0, int sh1, int sh2, float in[sh0][sh1][sh2], float *out) {
	int i, j, k, total = 0;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}


void dense(float *in, float *weights, float *out, int sh_in, int sh_out) {
	int i, j;
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j * sh_out + i];
		}
		out[i] = sum;
	}
}


void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}

void get_VGG16_predict(int only_convolution) {

	cl_int error;
    // Create device buffers.
    matrix = create_and_init_vector();
    kernel_ = create_and_init_vector();
    output = create_result_buffer();
    host_result = malloc(sizeof(float) * SIZE * SIZE);

    // Create kernel
    kernel = clCreateKernel(g_program, "convolution_3_x_3", &error);
    ocl_err(error);

	int level, cur_size;

	// Init intermediate memory
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);

	time_measure_start("Layer1");
	// Layer 1 (Convolution 3 -> 64)
	level = 0;
	cur_size = SIZE;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  image, wc[level], bc[level], mem_block1);
	time_measure_stop_and_print("Layer1");


	time_measure_start("Layer2");
	// Layer 2 (Convolution 64 -> 64)
	level = 1;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
					  time_measure_stop_and_print("Layer2");
	reset_mem_block(mem_block1);
	
	
	time_measure_start("Layer3");
	// Layer 3 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block2);
	cur_size /= 2;
	time_measure_stop_and_print("Layer3");
	
	time_measure_start("Layer4");
	// Layer 4 (Convolution 64 -> 128)
	level = 2;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer4");


	time_measure_start("Layer5");
	// Layer 5 (Convolution 128 -> 128)
	level = 3;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer5");
	
	time_measure_start("Layer6");
	// Layer 6 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block2);
	cur_size /= 2;
	time_measure_stop_and_print("Layer6");

	time_measure_start("Layer7");
	// Layer 7 (Convolution 128 -> 256)
	level = 4;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer7");

	time_measure_start("Layer8");
	// Layer 8 (Convolution 256 -> 256)
	level = 5;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer8");

	time_measure_start("Layer9");
	// Layer 9 (Convolution 256 -> 256)
	level = 6;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer9");
	
	time_measure_start("Layer10");
	// Layer 10 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block1);
	cur_size /= 2;
	time_measure_stop_and_print("Layer10");
	
	time_measure_start("Layer11");
	// Layer 11 (Convolution 256 -> 512)
	level = 7;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer11");

	time_measure_start("Layer12");
	// Layer 12 (Convolution 512 -> 512)
	level = 8;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer12");

	time_measure_start("Layer13");
	// Layer 13 (Convolution 512 -> 512)
	level = 9;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer13");
	
	time_measure_start("Layer14");
	// Layer 14 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block2);
	// for (i = 0; i < cshape[level][0]; i++) {
	// 	maxpooling(cur_size, &mem_block2[i * cur_size * cur_size]);
	// }
	cur_size /= 2;
	time_measure_stop_and_print("Layer14");
	
	time_measure_start("Layer15");
	// Layer 15 (Convolution 512 -> 512)
	level = 10;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer15");

	time_measure_start("Layer16");
	// Layer 16 (Convolution 512 -> 512)
	level = 11;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer16");

	time_measure_start("Layer17");
	// Layer 17 (Convolution 512 -> 512)
	level = 12;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer17");
	
	time_measure_start("Layer18");
	// Layer 18 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block1);
	cur_size /= 2;
	time_measure_stop_and_print("Layer18");
	
	mem_block1_dense = mem_block2;
	mem_block2_dense = mem_block1;

	reset_mem_block_dense(mem_block1_dense);
	time_measure_start("Layer19");
	// Layer 19 (Flatten)
	flatten(cshape[level][0], cur_size, cur_size, mem_block1, mem_block1_dense);
	time_measure_stop_and_print("Layer19");
	if (only_convolution == 1) {
		return;
	}

	//reset_mem_block_dense(mem_block2_dense);

	time_measure_start("Layer20");
	// Layer 20 (Dense)
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block1_dense);
	time_measure_stop_and_print("Layer20");

	time_measure_start("Layer21");
	// Layer 21 (Dense)
	level = 1;
	dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block2_dense);
	time_measure_stop_and_print("Layer21");
	
	time_measure_start("Layer22");
	// Layer 22 (Dense)
	level = 2;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	softmax(mem_block2_dense, dshape[level][1]);
	// dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);
	time_measure_stop_and_print("Layer22");
	
	return;
}


void output_predictions() {
	float max = 0.f;
	int max_idx = 0;
	for (int i = 0; i < dshape[2][1]; i++) {
		 if (max <  mem_block2_dense[i]) {
			 max = mem_block2_dense[i];
			 max_idx = i;
		 }
	}

	printf("Prediction: %s (score = %f)\n", imagenet_labels[max_idx], max);
}



int main(int argc, char *argv[]) {
	char buf[1024];
	char *weights_file;
	char *output_file;
	int lvls = -1;
	int only_convolution = 0;

	if (argc != 3) {
		printf("Usage: <program> <weights file> <image>\n");
		return 0;
	}
	weights_file = argv[1];
	if (argc == 5) {
		lvls = 13;
		only_convolution = 1;
	}

	cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

	init_memory();
	read_weights(weights_file, lvls);

	printf("Reads finished.");

	int width, height, num_channels;
    unsigned char *input_image = stbi_load(argv[2], &width, &height, &num_channels, 3);
	store_image(input_image);
	normalize_image();

	time_measure_start("prediction");
	get_VGG16_predict(only_convolution);
	time_measure_stop_and_print("prediction");

	output_predictions();

	free_memory();

	return 0;
}

