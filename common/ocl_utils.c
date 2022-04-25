#define DECLARE_GLOBALS
#include "ocl_utils.h"

#include <stdio.h>
#include <stdlib.h>

static const size_t NAME_SIZE = 256;

cl_platform_id ocl_select_platform(void)
{
    cl_uint num_platforms;
    ocl_err(clGetPlatformIDs(0, NULL, &num_platforms));

    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    ocl_err(clGetPlatformIDs(num_platforms, platforms, NULL));

    int ret;
    int index;
    do {
        printf("Select platform: \n");
        for (int i = 0; i < num_platforms; ++i)
        {
            char name[NAME_SIZE];
            ocl_err(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, NAME_SIZE, 
                        name, NULL));
            char vendor[NAME_SIZE];
            ocl_err(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, NAME_SIZE, 
                        vendor, NULL));
            char version[NAME_SIZE];
            ocl_err(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, NAME_SIZE, 
                        version, NULL));


            printf("\t - %d: %s %s, OpenCL version: %s\n", i + 1, vendor, name,
                    version);
        }
        if (num_platforms == 1)
        {
            printf("Using the only platform available.\n");
            cl_platform_id platform = platforms[0];
            free(platforms);
            return platform;
        }

        ret = scanf("%d%*c", &index);
    } while (index < 1 || index > num_platforms || ret != 1);


    cl_platform_id platform = platforms[index - 1];

    free(platforms);

    return platform;
}

cl_device_id ocl_select_device(cl_platform_id platform)
{
    cl_uint num_devices;
    ocl_err(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

    cl_device_id *devices =
        (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    ocl_err(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                devices, NULL));


    int ret;
    int index;
    do {
        printf("Select device: \n");
        for (int i = 0; i < num_devices; ++i)
        {
            char name[NAME_SIZE];
            ocl_err(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, NAME_SIZE,
                        name, NULL));

            printf("\t - %d: %s\n", i + 1, name);
        }

        if (num_devices == 1)
        {
            printf("Using the only device available.\n");
            cl_device_id device = devices[0];
            free(devices);
            g_device = device;
            return device;
        }

        ret = scanf("%d%*c", &index);
    } while (index < 1 || index > num_devices || ret != 1);

    cl_device_id device = devices[index - 1];

    free(devices);
    g_device = device;

    return device;
}

void init_ocl(cl_device_id device)
{
    cl_int error;
    g_context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    ocl_err(error);

    g_command_queue = clCreateCommandQueue(g_context, device, 0, &error);
    ocl_err(error);
}

void create_program(const char *source_name, const char *options)
{
    cl_int error;
    const char *source = read_source_file(source_name);
    g_program = clCreateProgramWithSource(g_context, 1, &source, NULL, &error);
    ocl_err(error);

    error = clBuildProgram(g_program, 1, &g_device, options, NULL, NULL);
    if (error)
    {
        fprintf(stderr, "ERROR: Kernel compilation failed.\n");
        char* build_log;
        size_t log_size;
        //
        // First call to know the proper size
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        build_log = malloc(sizeof(char)*(log_size+1));
        //
        // Second call to get the log
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("%s\n", build_log);
        ocl_err(error);
    }
    free(source);
}

char *read_source_file(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("Failed to read file");
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *code = (char *)malloc(file_size + 1);
    fread(code, file_size, 1, fp);
    fclose(fp);
    code[file_size] = 0;
    return code;
}

const char *errorString(const int err)
{
    const char* errString = NULL;

    switch (err)
    {
        case CL_SUCCESS:
            errString = "Success";
            break;
        case CL_DEVICE_NOT_FOUND:
            errString = "Device not found";
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            errString = "Device not available";
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            errString = "Compiler not available";
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            errString = "Memory object allocation failure";
            break;
        case CL_OUT_OF_RESOURCES:
            errString = "Out of resources";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            errString = "Out of host memory";
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            errString = "Profiling info not available";
            break;
        case CL_MEM_COPY_OVERLAP:
            errString = "Memory copy overlap";
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            errString = "Image format mismatch";
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            errString = "Image format not supported";
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            errString = "Build program failure";
            break;
        case CL_MAP_FAILURE:
            errString = "Map failure";
            break;
        case CL_INVALID_VALUE:
            errString = "Invalid value";
            break;
        case CL_INVALID_DEVICE_TYPE:
            errString = "Invalid device type";
            break;
        case CL_INVALID_PLATFORM:
            errString = "Invalid platform";
            break;
        case CL_INVALID_DEVICE:
            errString = "Invalid device";
            break;
        case CL_INVALID_CONTEXT:
            errString = "Invalid context";
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            errString = "Invalid queue properties";
            break;
        case CL_INVALID_COMMAND_QUEUE:
            errString = "Invalid command queue";
            break;
        case CL_INVALID_HOST_PTR:
            errString = "Invalid host pointer";
            break;
        case CL_INVALID_MEM_OBJECT:
            errString = "Invalid memory object";
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            errString = "Invalid image format descriptor";
            break;
        case CL_INVALID_IMAGE_SIZE:
            errString = "Invalid image size";
            break;
        case CL_INVALID_SAMPLER:
            errString = "Invalid sampler";
            break;
        case CL_INVALID_BINARY:
            errString = "Invalid binary";
            break;
        case CL_INVALID_BUILD_OPTIONS:
            errString = "Invalid build options";
            break;
        case CL_INVALID_PROGRAM:
            errString = "Invalid program";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            errString = "Invalid program executable";
            break;
        case CL_INVALID_KERNEL_NAME:
            errString = "Invalid kernel name";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            errString = "Invalid kernel definition";
            break;
        case CL_INVALID_KERNEL:
            errString = "Invalid kernel";
            break;
        case CL_INVALID_ARG_INDEX:
            errString = "Invalid argument index";
            break;
        case CL_INVALID_ARG_VALUE:
            errString = "Invalid argument value";
            break;
        case CL_INVALID_ARG_SIZE:
            errString = "Invalid argument size";
            break;
        case CL_INVALID_KERNEL_ARGS:
            errString = "Invalid kernel arguments";
            break;
        case CL_INVALID_WORK_DIMENSION:
            errString = "Invalid work dimension";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            errString = "Invalid work group size";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            errString = "invalid work item size";
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            errString = "Invalid global offset";
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            errString = "Invalid event wait list";
            break;
        case CL_INVALID_EVENT:
            errString = "Invalid event";
            break;
        case CL_INVALID_OPERATION:
            errString = "Invalid operation";
            break;
        case CL_INVALID_GL_OBJECT:
            errString = "Invalid OpenGL object";
            break;
        case CL_INVALID_BUFFER_SIZE:
            errString = "Invalid buffer size";
            break;
        case CL_INVALID_MIP_LEVEL:
            errString = "Invalid MIP level";
            break;
        default:
            errString = "No error string found, use Google";
    }
    return errString;
}
