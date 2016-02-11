#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRenderPointsAndNormals
{
public:
	static const std::string kernel_name;

    KernelRenderPointsAndNormals(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume_d,
            VolumeBuffer & volume_dw,
            ImageBuffer & rendered_mask,
            ImageBuffer & rendered_points,
            ImageBuffer & rendered_normals,
            float voxel_size,
            Eigen::Affine3f const& model_pose,
            Eigen::Array2f const& camera_focal,
            Eigen::Array2f const& camera_center,
            float min_render_depth,
            float max_render_depth,
            int mask_value
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
