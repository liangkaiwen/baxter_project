#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRaytraceSpecial
{
public:
	static const std::string kernel_name;

    KernelRaytraceSpecial(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            ImageBuffer & rendered_mask,
            ImageBuffer & rendered_points,
            ImageBuffer & rendered_normals,
            const Eigen::Affine3f & model_pose,
            const Eigen::Affine3f & object_pose,
            const Eigen::Array2f & camera_focal,
            const Eigen::Array2f & camera_center,
            float min_render_depth,
            float max_render_depth,
            int mask_value
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
