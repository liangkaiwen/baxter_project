#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

class KernelPointsToDepthImage
{
public:
	static const std::string kernel_name;

    KernelPointsToDepthImage(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            ImageBuffer & points,
            ImageBuffer & result_depth_image
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
