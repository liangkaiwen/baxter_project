#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelAddVolumes
{
public:
	static const std::string kernel_name;

    KernelAddVolumes(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume_1,
            VolumeBuffer & volume_counts_1,
            VolumeBuffer & volume_2,
            VolumeBuffer & volume_counts_2,
            VolumeBuffer & result_volume,
            VolumeBuffer & result_counts
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
