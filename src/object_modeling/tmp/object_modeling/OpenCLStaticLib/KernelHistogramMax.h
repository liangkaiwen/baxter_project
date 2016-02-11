#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelHistogramMax
{
public:
    static const std::string kernel_name;

    KernelHistogramMax(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & buffer_histogram_bin,
            VolumeBuffer & buffer_index,
            VolumeBuffer & buffer_value,
            int index
            );

protected:
    CL & cl_;
    cl::Kernel kernel_;
};
