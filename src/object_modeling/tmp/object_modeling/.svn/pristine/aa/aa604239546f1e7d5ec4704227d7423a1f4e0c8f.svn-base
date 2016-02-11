#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <vector>

// this bit of crap is attempting to deal with OpenCL 1.1 only support from Nvidia, but Opencl 1.2 headers...
// it didn't work...instead, I manually overwrote cl.hpp with the OpenCL 1.1 version
#if 0
#include "CL/cl.h"
//#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#undef CL_VERSION_1_2
#endif



// I believe this is the only place this is included
#pragma warning( push )
#pragma warning( disable : 4290 )
#include "CL/cl.hpp"
#pragma warning( pop )

enum OpenCLPlatformType {
    OPENCL_PLATFORM_NVIDIA = 0,
    OPENCL_PLATFORM_INTEL = 1,
    OPENCL_PLATFORM_AMD = 2};

enum OpenCLContextType {
    OPENCL_CONTEXT_DEFAULT,
    OPENCL_CONTEXT_CPU,
    OPENCL_CONTEXT_GPU};

class CL {
    public:
        CL(OpenCLPlatformType platform_type, OpenCLContextType context_type);

        bool isInitialized() const {return is_initialized;}

        // returns the index into programs (or -1 on error)
        int loadProgram(std::string program_source, std::string source_full_path = "",  std::string additional_args = "");

//    private:
        OpenCLPlatformType platform_type;
        OpenCLContextType context_type;

        bool is_initialized;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue queue;
        std::vector<cl::Program> programs;
        //cl::Kernel kernel;

        cl_int err;
        // Do I need events?
        //cl::Event event;
};
