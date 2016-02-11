#pragma once

#include "OpenCLAllKernels.h"

class KernelApplyPoseToPoints
{
public:
	static const std::string kernel_name;

    KernelApplyPoseToPoints(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & buffer, Eigen::Affine3f const& pose, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
