#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelComputeNormalVolumeWithWeightsUnnormalized
{
public:
	static const std::string kernel_name;

    KernelComputeNormalVolumeWithWeightsUnnormalized(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume,
            VolumeBuffer & weights,
            VolumeBuffer & result
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
