#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelComputeNormalVolumeWithWeights
{
public:
	static const std::string kernel_name;

    KernelComputeNormalVolumeWithWeights(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume,
            VolumeBuffer & weights,
            VolumeBuffer & result
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
