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

#include "common/time_utils.h"
#include "common/ocl_utils.h"
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
cl_kernel zerokernel;
cl_mem layer;
cl_mem zeropad;
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
	memset(mem, 0, MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(cl_float));
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
	mem_block1 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(cl_float));
	mem_block2 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(cl_float));
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

cl_mem create_and_init_vector(int size)
{
    cl_int error;
    cl_mem dev_vec = clCreateBuffer(g_context,
            CL_MEM_READ_WRITE,
            sizeof(float) * size * size * MEM_BLOCK_DEPTH, NULL, &error);
    ocl_err(error);
	printf("created dev vector\n");
    return dev_vec;
}

cl_mem create_result_buffer(void)
{
    cl_int error;
    cl_mem dev_vec = clCreateBuffer(g_context, CL_MEM_READ_WRITE,
            sizeof(float) * SIZE * SIZE * MEM_BLOCK_DEPTH, NULL, &error);
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
void gpu_shenanigans(int size, float matrix[][size], float kernel[][CONV_SIZE],
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

void testFunctionZeropad()
{
	printf("testFunctionZeropad() called");
	cl_int error;
	
	cl_mem test_fill = clCreateBuffer(g_context,
            CL_MEM_READ_WRITE,
            sizeof(float) * SIZE * SIZE * MEM_BLOCK_DEPTH, NULL, &error);

	//zeropad in convolution_layer() function
	cl_mem test_zero = clCreateBuffer(g_context,
            CL_MEM_READ_WRITE,
            sizeof(float) * (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH, NULL, &error);

	float zero[] = {0.0};
	ocl_err(clEnqueueFillBuffer(g_command_queue, test_zero, zero, 1, 0, sizeof(cl_float) * (SIZE + 2) * (SIZE + 2) * MEM_BLOCK_DEPTH, 0, NULL, NULL));
	// maak buffer aan van SIZE op SIZE op MEM_BLOCK_DEPTH om data in te vullen

	float* test = malloc(sizeof(float) * SIZE * SIZE * MEM_BLOCK_DEPTH);
	// test is 1 dimensionale array die SIZE op SIZE op MEM_BLOCK_DEPTH groot is
	for (int i = 0; i < MEM_BLOCK_DEPTH; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			for (int k = 0; k < SIZE; k++)
			{
				test[i * (SIZE * SIZE) + j * (SIZE) + k] = k;
				// vul matrix test met random getalletjes
			}
		}
	}

	ocl_err(clEnqueueWriteBuffer(g_command_queue, test_fill, CL_TRUE, 0, sizeof(cl_float) * SIZE * SIZE * MEM_BLOCK_DEPTH, test, 0, NULL, NULL));
	// vul buffer met alle informatie van de matrix test

	cl_kernel test_kernel = clCreateKernel(g_program, "zeroshit", &error);
	// maak kernel aan die zeroshit gaat uitvoeren

	// maak argumenten aan voor zeroshit kernel
	int feature_size = SIZE;

	// geef correcte argumenten mee aan kernel
	int arg_num = 0;
	ocl_err(clSetKernelArg(test_kernel, arg_num++, sizeof(cl_int), &feature_size));
	ocl_err(clSetKernelArg(test_kernel, arg_num++, sizeof(cl_mem), &test_zero));
	ocl_err(clSetKernelArg(test_kernel, arg_num++, sizeof(cl_mem), &test_fill));

	// voer kernel uit en deel GPU op in correcte delen
	size_t work_sizes[] = {SIZE, SIZE, MEM_BLOCK_DEPTH};
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, test_kernel, 3, NULL, work_sizes, NULL, 0, NULL, NULL));

	// wacht tot kernels gedaan zijn
	ocl_err(clFinish(g_command_queue));

	FILE *fp;
	fp = fopen("zeropadtest.txt", "w");
	
	// maak array aan en uitkomst van kernel
	float* test_results = malloc(sizeof(float) * (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH);
	ocl_err(clEnqueueReadBuffer(g_command_queue, test_zero, CL_TRUE, 0, sizeof(cl_float) * (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH, test_results, 0, NULL, NULL));
	printf("Test\n");
	int x = 0;
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < SIZE+2; j++)
		{
			for (int k = 0; k < SIZE+2; k++)
			{
				// print eerste vlak af van test_Results om te testen of hij ingevuld word.
				//if (test_results[i * ((SIZE+2) * (SIZE+2)) + j * (SIZE+2) + k] != 0){
					printf("%f\n", test_results[i * ((SIZE+2) * (SIZE+2)) + j * (SIZE+2) + k]);
					fprintf(fp, "%f\n", test_results[i * ((SIZE+2) * (SIZE+2)) + j * (SIZE+2) + k]);
				//}
			}
		}
	}
	fclose(fp);
	printf("testFunctionZeropad() finished\n");
}


void convolution_layer(int feature_size, int input_depth, int output_depth,
					   float *input_features, float *layer_weights, float *layer_biases, float *output_features, int level) // momenteel worden alle lagen van de input meegegeven en niet maar 1 laag, dit moet nog aangepast worden
{
	// we geven hierbij een hele laag (in diepte mee) en deze moet de gpu dan verwerken
	// dit zorgt voor minder communicatie tussen gpu en cpu en daardoor (hopelijk) een 
	// kortere compute tijd.

	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*																ZEROKERNEL "zeroshit"															*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	float zero[] = {0.0};
	/* Zeropad buffer 
	 *	- (2 px bigger in width and height)
	 *	- Size of (SIZE + 2)*(SIZE + 2)*MEM_BLOCK_DEPTH
	 */
	ocl_err(clEnqueueFillBuffer(g_command_queue, zeropad, zero, 1, 0, sizeof(cl_float) * (SIZE + 2) * (SIZE + 2) * input_depth, 0, NULL, NULL));
	// schrijf zeropad buffer vol met 0 (ter grootte van heel de zeropad buffer (float * (size+2)(size+2)(input_depth)))

	ocl_err(clEnqueueWriteBuffer(g_command_queue, matrix,  CL_TRUE, 0, sizeof(cl_float) * SIZE * SIZE * input_depth, input_features, 0, NULL, NULL));
	// schrijft data weg in buffer 'matrix' van input features (ter grootte van (SIZE * SIZE * input_depth))

	cl_uint arg_num = 0;
	ocl_err(clSetKernelArg(zerokernel, arg_num++, sizeof(cl_int), &feature_size));
	ocl_err(clSetKernelArg(zerokernel, arg_num++, sizeof(cl_mem), &zeropad));
	ocl_err(clSetKernelArg(zerokernel, arg_num++, sizeof(cl_mem), &matrix));
	// geef de waarde van feature size door aan de kernel
	// geef de buffers zeropad en matrix door aan kernel

	size_t work_sizes1[] = {feature_size, feature_size, input_depth};
	// deel de gpu in in feature_size * feature_size * input_depth work_items op om te parralelliseren

	ocl_err(clEnqueueNDRangeKernel(g_command_queue, zerokernel, 3, NULL, work_sizes1, NULL, 0, NULL, NULL));
	// voer de zerokernel uit

	float *z = malloc(sizeof(float) * MEM_BLOCK_DEPTH * (SIZE+2) * (SIZE+2));
	memset(z, 0.0, (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH);
	ocl_err(clFinish(g_command_queue));
	// wacht tot alle zerokernels gedaan zijn
	
	ocl_err(clEnqueueReadBuffer(g_command_queue, zeropad, CL_TRUE, 0, sizeof(cl_float) * (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH, z, 0, NULL, NULL));
	// lees waarde van zeropad-buffer in en steek deze in z (ter grootte van (float * (SIZE+2) * (SIZE+2) * MEM_BLOCK_DEPTH)

	// FILE *fp1;
	// FILE *fp2;
	// fp1 = fopen("zeropad_without_zeroes.txt", "w");
	// fp2 = fopen("zeropad_full.txt", "w");

	// int x = 0;
	// for (int i = 0; i<1; i++)
	// {
	// 	for(int j = 0; j < SIZE+2;j++)
	// 	{
	// 		for (int k = 0; k < SIZE+2; k++)
	// 		{
	// 			if (z[i * (SIZE+2) * (SIZE+2) + j * (SIZE+2) + k] != 0.0){
	// 				x += 1;
	// 				//printf("%f\n", z[i * (SIZE+2) * (SIZE+2) + j * (SIZE+2) + k]);
	// 				fprintf(fp1, "%f\n", z[i * (SIZE+2) * (SIZE+2) + j * (SIZE+2) + k]);
	// 			// zeropad kan niet beschreven worden want hij toont allemaal
	// 			}
	// 			fprintf(fp2, "%f\n", z[i * (SIZE+2) * (SIZE+2) + j * (SIZE+2) + k]);
	// 		}
	// 		// printf("]\n");
	// 	}
	// 	// printf("]\n");
	// }
	// fclose(fp1);
	// fclose(fp2);
	//printf("niet lege plekken: %d\n", x);
	// exit(0);

	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*																	KERNEL "gpu_shenanigans"													*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------------------------------------------------------------------------------------------------------------------*/
	// vul buffer matrix en kernel_ in met de data van input_features en layer_weights
	// printf("writing to matrix\n");
	//clEnqueueWriteBuffer(g_command_queue, matrix,  CL_TRUE, 0, sizeof(cl_float) * feature_size * feature_size * input_depth, input_features, 0, NULL, NULL);
	// printf("wrote to matrix\n");

	clEnqueueWriteBuffer(g_command_queue, kernel_, CL_TRUE, 0, sizeof(cl_float) * output_depth * input_depth * CONV_SIZE * CONV_SIZE, layer_weights , 0, NULL, NULL);
	// vul buffer kernel_ met data van layer_weights (ter grootte van cl_float * output_depth * input_depth * CONV_SIZE * CONV_SIZE)

	clEnqueueWriteBuffer(g_command_queue, layer  , CL_TRUE, 0, sizeof(cl_float) * output_depth, layer_biases , 0, NULL, NULL);
	// vul buffer layer in met data van layer_biases (ter grootte van float * output_depth)

	// geef alle nodige argumenten mee aan de gpu (hier moeten er nog extra bij om te weten welke laag er effectief nodig is)
	arg_num = 0;
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &feature_size));
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &input_depth));
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &output_depth));
	// ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &matrix)); // input_features moet niet meer meegegeven worden wegens zeropad al ingevuld
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &kernel_));// layer_weights
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &layer));// layer_biases
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &output)); // output_features
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &zeropad)); // zeropad
	// geef argumenten mee met kernel: feature_size, input_depth, output_depth als gewone int
	// geef buffers mee aan kernel: kernel_, layer, output, zeropad (telkens een pointer naar het eerste element)

	size_t work_sizes2[] = {feature_size, feature_size, input_depth};
	// deel gpu work_items op in feature_size * feature_size * input_depth onderdelen
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 3, NULL, work_sizes2, NULL, 0, NULL, NULL));
	// laat de gpu al zijn kernels uitvoeren

	ocl_err(clFinish(g_command_queue));
	// wacht tot alle kernels op de gpu gedaan zijn

	ocl_err(clEnqueueReadBuffer(g_command_queue, output, CL_TRUE, 0, sizeof(cl_float) * SIZE * SIZE * output_depth, output_features, 0, NULL, NULL));
	// lees van de output-buffer van output in output_features



	for (int i = 0; i < output_depth; i++)
	{
		// add_bias_and_relu(feature_size, &output_features[i * feature_size * feature_size], layer_biases[i]);
	}

	// printf("read output from GPU\n");
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
    matrix = create_and_init_vector(SIZE);
    kernel_ = create_and_init_vector(SIZE);
	layer = create_and_init_vector(SIZE);
	zeropad = create_and_init_vector(SIZE + 2);
    output = create_result_buffer();
    host_result = malloc(sizeof(float) * SIZE * SIZE);

    // Create kernel
    kernel = clCreateKernel(g_program, "gpu_shenanigans", &error);
	zerokernel = clCreateKernel(g_program, "zeroshit", &error);
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
					  image, wc[level], bc[level], mem_block1, level);
	time_measure_stop_and_print("Layer1");

	
	time_measure_start("Layer2");
	// Layer 2 (Convolution 64 -> 64)
	level = 1;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2, level);
					  time_measure_stop_and_print("Layer2");
	reset_mem_block(mem_block1);
	
	memblock_to_file(cur_size * cshape[level][0] * cshape[level][0]);
	// exit(0);
	
	time_measure_start("Layer3");
	// Layer 3 (MaxPooling)
	maxpooling(cur_size, cshape[level][0], mem_block2);
	cur_size /= 2;
	time_measure_stop_and_print("Layer3");
	
	time_measure_start("Layer4");
	// Layer 4 (Convolution 64 -> 128)
	level = 2;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1, level);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer4");



	time_measure_start("Layer5");
	// Layer 5 (Convolution 128 -> 128)
	level = 3;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2, level);
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
					  mem_block2, wc[level], bc[level], mem_block1, level);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer7");

	time_measure_start("Layer8");
	// Layer 8 (Convolution 256 -> 256)
	level = 5;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2, level);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer8");

	time_measure_start("Layer9");
	// Layer 9 (Convolution 256 -> 256)
	level = 6;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1, level);
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
					  mem_block1, wc[level], bc[level], mem_block2, level);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer11");

	time_measure_start("Layer12");
	// Layer 12 (Convolution 512 -> 512)
	level = 8;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1, level);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer12");

	time_measure_start("Layer13");
	// Layer 13 (Convolution 512 -> 512)
	level = 9;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2, level);
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
					  mem_block2, wc[level], bc[level], mem_block1, level);
	reset_mem_block(mem_block2);
	time_measure_stop_and_print("Layer15");

	time_measure_start("Layer16");
	// Layer 16 (Convolution 512 -> 512)
	level = 11;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2, level);
	reset_mem_block(mem_block1);
	time_measure_stop_and_print("Layer16");

	time_measure_start("Layer17");
	// Layer 17 (Convolution 512 -> 512)
	level = 12;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1, level);
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
	//dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);
	time_measure_stop_and_print("Layer22");
	
	return;
}


void output_predictions() {
	float max = 0.f;
	int max_idx = 0;
	FILE *fp;
	fp = fopen("dense_result.txt", "w");
	for (int i = 0; i < dshape[2][1]; i++) {
		 if (max <  mem_block2_dense[i]) {
			 max = mem_block2_dense[i];
			 max_idx = i;
		 }
		fprintf(fp,"%f\n", mem_block2_dense[i]);
	}
	fclose(fp);
	printf("Prediction: %s (score = %f)\n", imagenet_labels[max_idx], max);
}

void memblock_to_file(int size){

	FILE *fp;
	fp = fopen("memblock.txt", "w");
	for (int i = 0; i < size; i++){
		fprintf(fp, "%f\n", mem_block2[i]);
	}
	fclose(fp);
}

int main(int argc, char *argv[]) {
	cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

	//testFunctionZeropad();
	//exit(1);

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

