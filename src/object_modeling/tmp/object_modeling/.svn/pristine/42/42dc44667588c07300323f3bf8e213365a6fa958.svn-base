#include "KernelRender2MeansAbs.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelRender2MeansAbs::kernel_name = "KernelRender2MeansAbs";

KernelRender2MeansAbs::KernelRender2MeansAbs(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


#if 0
KernelRender2MeansAbs(
	__global float *D_1,
	__global float *DW_1,
	__global float *D_2,
	__global float *DW_2,
	__global int *rendered_mask,
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose,
	const float16 inverse_pose,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int2 image_dims,
	const float min_render_depth,
	const float max_render_depth,
	const int replace_only_if_nearer,
	const int mask_value
	)
#endif


void KernelRender2MeansAbs::runKernel(
        VolumeBuffer & volume_1,
        VolumeBuffer & volume_weights_1,
		VolumeBuffer & volume_2,
        VolumeBuffer & volume_weights_2,
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
        )
{
    try {
		cl_int4 cl_volume_dims = {volume_1.getVolumeCellCounts()[0], volume_1.getVolumeCellCounts()[1], volume_1.getVolumeCellCounts()[2], 0};

        cl_float16 cl_pose = getCLPose(model_pose);
        cl_float16 cl_pose_inverse = getCLPose(model_pose.inverse());

        cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
        cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
        cl_int2 cl_camera_size = {rendered_mask.getCols(), rendered_mask.getRows()};

        cl_int replace_only_if_nearer = true;


        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_weights_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_2.getBuffer());
        kernel_.setArg(kernel_arg++, volume_weights_2.getBuffer());
        kernel_.setArg(kernel_arg++, rendered_mask.getBuffer());
        kernel_.setArg(kernel_arg++, rendered_points.getBuffer());
        kernel_.setArg(kernel_arg++, rendered_normals.getBuffer());
        kernel_.setArg(kernel_arg++, cl_volume_dims);
        kernel_.setArg(kernel_arg++, (cl_float)voxel_size);
        kernel_.setArg(kernel_arg++, cl_pose);
        kernel_.setArg(kernel_arg++, cl_pose_inverse);
        kernel_.setArg(kernel_arg++, cl_camera_focal);
        kernel_.setArg(kernel_arg++, cl_camera_center);
        kernel_.setArg(kernel_arg++, cl_camera_size);
        kernel_.setArg(kernel_arg++, (cl_float)min_render_depth);
        kernel_.setArg(kernel_arg++, (cl_float)max_render_depth);
        kernel_.setArg(kernel_arg++, (cl_int)replace_only_if_nearer);
        kernel_.setArg(kernel_arg++, (cl_int)mask_value);

        // run kernel
        cl::NDRange global(cl_camera_size.s[0], cl_camera_size.s[1]);
        cl::NDRange local = cl::NullRange;
        cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
