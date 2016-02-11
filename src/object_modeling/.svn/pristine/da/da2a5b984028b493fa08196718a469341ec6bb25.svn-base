#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

class KernelNormalsToShadedImage
{
public:
	static const std::string kernel_name;

    KernelNormalsToShadedImage(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            ImageBuffer & normals,
            ImageBuffer & result_image,
            Eigen::Vector3f const& vector_to_light
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
