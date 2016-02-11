#include "KernelOptimizeErrorAndJacobianImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelOptimizeErrorAndJacobianImage::kernel_name = "KernelOptimizeErrorAndJacobianImage";

KernelOptimizeErrorAndJacobianImage::KernelOptimizeErrorAndJacobianImage(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


#if 0
#define DEBUG_CODE_RENDER_INVALID 1;
#define DEBUG_CODE_PROJECT_OOB 2;
#define DEBUG_CODE_FRAME_INVALID 3;
#define DEBUG_CODE_OUTLIER_DISTANCE 4;
#define DEBUG_CODE_OUTLIER_ANGLE 5;
#define DEBUG_CODE_SUCCESS 6;

__kernel void KernelOptimizeErrorAndJacobianImage(
    __global float4 *render_points,
    __global float4 *render_normals,
    __global float *render_image,
    __global float4 *frame_points,
    __global float4 *frame_normals,
    __global float *frame_image,
    __global float *frame_image_gradient_x,
    __global float *frame_image_gradient_y,
    __global float *result_weighted_error, // (0 1 2 3 ...)
    __global float *result_weighted_jacobian, // (a0 a1 .. a6, b0 b1... b6, ...)
    __global int *result_debug_code,
    __global float *result_debug_huber_weight,
    __global float *result_debug_distance_weight,
    const int2 image_dims_render,
    const float2 camera_focal_render,
    const float2 camera_center_render,
    const int2 image_dims_frame,
    const float2 camera_focal_frame,
    const float2 camera_center_frame,
    const float16 x_transform,
    const float outlier_max_distance, // really?
    const float outlier_min_normal_dot_product, // really?
    const float huber_param_icp,
    const float huber_param_image
)
#endif


void KernelOptimizeErrorAndJacobianImage::runKernel(
        const ImageBuffer & render_points,
        const ImageBuffer & render_normals,
        const ImageBuffer & render_image,
        const ImageBuffer & frame_points,
        const ImageBuffer & frame_normals,
        const ImageBuffer & frame_image,
        const ImageBuffer & frame_image_gradient_x,
        const ImageBuffer & frame_image_gradient_y,
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
        const float huber_param_icp,
        const float huber_param_image
        )
{
	try {
        const int cols = render_points.getCols();
        const int rows = render_points.getRows();
		const int point_count = rows * cols;

		cl_float2 cl_camera_focal_render = {camera_focal_render[0], camera_focal_render[1]};
		cl_float2 cl_camera_center_render = {camera_center_render[0], camera_center_render[1]};
		cl_int2 cl_camera_size_render = {render_points.getCols(), render_points.getRows()};
		cl_float2 cl_camera_focal_frame = {camera_focal_frame[0], camera_focal_frame[1]};
		cl_float2 cl_camera_center_frame = {camera_center_frame[0], camera_center_frame[1]};
		cl_int2 cl_camera_size_frame = {frame_points.getCols(), frame_points.getRows()};

		cl_float16 cl_x_transform = getCLPose(x_transform);

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, render_points.getBuffer());
		kernel_.setArg(kernel_arg++, render_normals.getBuffer());
        kernel_.setArg(kernel_arg++, render_image.getBuffer());
		kernel_.setArg(kernel_arg++, frame_points.getBuffer());
		kernel_.setArg(kernel_arg++, frame_normals.getBuffer());
        kernel_.setArg(kernel_arg++, frame_image.getBuffer());
        kernel_.setArg(kernel_arg++, frame_image_gradient_x.getBuffer());
        kernel_.setArg(kernel_arg++, frame_image_gradient_y.getBuffer());
		kernel_.setArg(kernel_arg++, result_weighted_error.getBuffer());
		kernel_.setArg(kernel_arg++, result_weighted_jacobian.getBuffer());
		kernel_.setArg(kernel_arg++, result_debug_code.getBuffer());
		kernel_.setArg(kernel_arg++, result_debug_huber_weight.getBuffer());
		kernel_.setArg(kernel_arg++, result_debug_distance_weight.getBuffer());
		kernel_.setArg(kernel_arg++, cl_camera_size_render);
		kernel_.setArg(kernel_arg++, cl_camera_focal_render);
		kernel_.setArg(kernel_arg++, cl_camera_center_render);
		kernel_.setArg(kernel_arg++, cl_camera_size_frame);
		kernel_.setArg(kernel_arg++, cl_camera_focal_frame);
		kernel_.setArg(kernel_arg++, cl_camera_center_frame);
		kernel_.setArg(kernel_arg++, cl_x_transform);
		kernel_.setArg(kernel_arg++, outlier_max_distance);
		kernel_.setArg(kernel_arg++, outlier_min_normal_dot_product);
        kernel_.setArg(kernel_arg++, huber_param_icp);
        kernel_.setArg(kernel_arg++, huber_param_image);

		// run kernel
		cl::NDRange global(cl_camera_size_render.s[0], cl_camera_size_render.s[1]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
