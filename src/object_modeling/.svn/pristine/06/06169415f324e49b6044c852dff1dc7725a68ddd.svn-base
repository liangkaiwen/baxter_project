#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

class KernelNormalsToColorImage
{
public:
	static const std::string kernel_name;

    KernelNormalsToColorImage(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            ImageBuffer & normals,
            ImageBuffer & result_image
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
