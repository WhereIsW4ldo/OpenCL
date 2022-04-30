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

inline void AtomicAdd(volatile __global float *source, const float operand) 
{
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal,
                newVal.intVal) != prevVal.intVal);
}

void conv_3x3 (int size, __global float *matrix, __global float *_kernel, __global float *out, __global float *zeropad)
{
	int i, j;
	float sum;

	i = get_global_id(0);
	j = get_global_id(1);	
	
	sum = zeropad[i*size + j] * _kernel[0 * 3 + 0] +
		zeropad[(i + 1)*size + j] * _kernel[1 * 3 + 0] +
		zeropad[(i + 2) * size + j] * _kernel[2 * 3 + 0]+
		zeropad[i*size + j + 1] * _kernel[0 * 3 + 1]+
		zeropad[(i + 1) * size + j + 1] * _kernel[1 * 3 + 1]+
		zeropad[(i + 2) * size + j + 1] * _kernel[2 * 3 + 1]+
		zeropad[i * size + j + 2] * _kernel[0 * 3 + 2]+
		zeropad[(i + 1) * size + j + 2] * _kernel[1 * 3 + 2]+
		zeropad[(i + 2) * size + j + 2] * _kernel[2 * 3 + 2];
	
	AtomicAdd(&out[(size*i) + j], sum);
	
}

void add_bias_and_relu(int size, __global float *out, float bs) {
	int id_x = get_global_id(0);
	int id_y = get_global_id(1);

	AtomicAdd(&out[(id_x * size) + id_y], bs); 
	if (out[(id_x * size) + id_y] < 0)
		out[(id_x * size) + id_y] = 0.0;

}

__kernel void gpu_shenanigans(
    int feature_size, int input_depth, int output_depth, __global float *input_features, 
    __global float *layer_weights, __global float *layer_biases, __global float *output_features, __global float *zeropad)
{

	int id_x = get_global_id(0);
	int id_y = get_global_id(1);
	int id_z = get_global_id(2);

	zeropad[id_z * (SIZE+2) * (SIZE+2) + (id_y + 1) * (SIZE+2) + id_x + 1] = input_features[id_z * (feature_size) * (feature_size) + id_y * (feature_size) + id_x];
	
	// vul 3D zeropad in met input_features data
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (int output_it = 0; output_it < output_depth; output_it++) 
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		conv_3x3(feature_size, &input_features[id_z * feature_size * feature_size],
							&layer_weights[output_it * input_depth * CONV_SIZE * CONV_SIZE +
											id_z * CONV_SIZE * CONV_SIZE],
							&output_features[output_it * feature_size * feature_size], &zeropad[id_z * (SIZE+2) * (SIZE+2)]);
		
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // om te synchroniseren tussen alle work-units
		if (id_z == 0)
			add_bias_and_relu(feature_size, &output_features[output_it * feature_size * feature_size], layer_biases[output_it]);
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}

/*
 * A simple kernel that "should" fill in buffer 'zeropad' with values from 'data'.
 * This zeropad has a padding of 1 width around all data from 'data'.
 */
__kernel void zeroshit(
	int feature_size, __global float *zeropad, __global float *data
)
{
	int id_x = get_global_id(0); // loopt van 0 -> feature_size
	int id_y = get_global_id(1); // loopt van 0 -> feature_size
	int id_z = get_global_id(2); // loopt van 0 -> input_depth

	zeropad[id_z * (SIZE+2) * (SIZE+2) + (id_y + 1) * (SIZE+2) + id_x + 1] = data[id_z * (feature_size) * (feature_size) + id_y * (feature_size) + id_x];
}