#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelDotVolumeNormal
{
public:
	static const std::string kernel_name;

    KernelDotVolumeNormal(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume,
            VolumeBuffer & result,
            Eigen::Vector3f & vector
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
