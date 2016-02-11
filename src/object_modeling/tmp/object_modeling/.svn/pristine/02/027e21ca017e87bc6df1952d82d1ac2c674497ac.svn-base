#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelVignetteApplyModelPolynomial3Uchar4
{
public:
	static const std::string kernel_name;

    KernelVignetteApplyModelPolynomial3Uchar4(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            const ImageBuffer & input_image,
            ImageBuffer & output_image,
            const Eigen::Array2f & camera_center,
            const Eigen::Array3f & vignette_model
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
