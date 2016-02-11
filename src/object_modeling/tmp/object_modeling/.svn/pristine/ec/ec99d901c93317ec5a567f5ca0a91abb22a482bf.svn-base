#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelOptimizeErrorAndJacobianICP
{
public:
	static const std::string kernel_name;

    KernelOptimizeErrorAndJacobianICP(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            const ImageBuffer & render_points,
            const ImageBuffer & render_normals,
            const ImageBuffer & frame_points,
            const ImageBuffer & frame_normals,
            ImageBuffer & result_weighted_error,
            ImageBuffer & result_weighted_jacobian,
            ImageBuffer & result_debug_code,
            ImageBuffer & result_debug_huber_weight,
            ImageBuffer & result_debug_distance_weight,
            const Eigen::Array2f & camera_focal_render,
            const Eigen::Array2f & camera_center_render,
            const Eigen::Array2f & camera_focal_frame,
            const Eigen::Array2f & camera_center_frame,
            const Eigen::Affine3f & x_transform,
            const float outlier_max_distance,
            const float outlier_min_normal_dot_product,
            const float huber_param_icp
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
