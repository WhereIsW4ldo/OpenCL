#ifndef _OCL_UTILS_H_
#define _OCL_UTILS_H_

#ifdef __APPLE__
#   include <OpenCL/opencl.h>
#else
#   include <CL/cl.h>
#endif

#ifndef DECLARE_GLOBALS
#define EXTERNAL extern
#else
#define EXTERNAL
#endif

const char *errorString(const int err);
char *read_source_file(const char *filename);

/// Interactively ask for the desired opencl platform
cl_platform_id ocl_select_platform(void);
//
/// Interactively ask for the desired opencl device
cl_device_id ocl_select_device(cl_platform_id platform);

/// 
void init_ocl(cl_device_id device);
void create_program(const char *source_name, const char *options);

EXTERNAL cl_command_queue g_command_queue;
EXTERNAL cl_context g_context;
EXTERNAL cl_program g_program;
EXTERNAL cl_device_id g_device;

#define ocl_err(code) \
do {\
    if ((code) != CL_SUCCESS)\
        {\
        fprintf(stderr, "OpenCL ERROR:\n\t"\
                        "Code: %d\n\t"\
                        "Problem: %s\n\t"\
                        "Filename: %s\n\t"\
                        "Line: %d\n", (code),\
                        errorString(code),\
                        __FILE__, __LINE__);\
        abort();\
    }\
} while (0)

#endif // __OCL_UTILS_H_
