#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelHistogramMaxCheckIndex
{
public:
    static const std::string kernel_name;

    KernelHistogramMaxCheckIndex(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & buffer_histogram_bin,
            VolumeBuffer & buffer_previous_index,
            VolumeBuffer & buffer_index,
            VolumeBuffer & buffer_value,
            int index,
            int index_range
            );

protected:
    CL & cl_;
    cl::Kernel kernel_;
};
