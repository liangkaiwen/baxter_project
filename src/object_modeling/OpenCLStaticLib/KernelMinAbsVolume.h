#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelMinAbsVolume
{
public:
	static const std::string kernel_name;

    KernelMinAbsVolume(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume_1,
            VolumeBuffer & volume_counts_1,
            VolumeBuffer & volume_2,
            VolumeBuffer & volume_counts_2,
            VolumeBuffer & result_volume,
            VolumeBuffer & result_counts,
            float minimum_relative_count);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
