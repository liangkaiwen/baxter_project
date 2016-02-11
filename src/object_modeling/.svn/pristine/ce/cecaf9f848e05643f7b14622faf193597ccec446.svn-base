#include "KernelRenderPoints.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelRenderPoints::kernel_name = "KernelRenderPoints";

KernelRenderPoints::KernelRenderPoints(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelRenderPoints::runKernel(
        VolumeBuffer & volume_d,
        VolumeBuffer & volume_dw,
        ImageBuffer & rendered_mask,
        ImageBuffer & rendered_points,
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
        cl_int4 cl_volume_dims = {volume_d.getVolumeCellCounts()[0], volume_d.getVolumeCellCounts()[1], volume_d.getVolumeCellCounts()[2], 0};

        cl_float16 cl_model_pose = getCLPose(model_pose);
        cl_float16 cl_model_pose_inverse = getCLPose(model_pose.inverse());

        cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
        cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
        cl_int2 cl_camera_size = {rendered_mask.getCols(), rendered_mask.getRows()};

        cl_int replace_only_if_nearer = true;


        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume_d.getBuffer());
        kernel_.setArg(kernel_arg++, volume_dw.getBuffer());
        kernel_.setArg(kernel_arg++, rendered_mask.getBuffer());
        kernel_.setArg(kernel_arg++, rendered_points.getBuffer());
        kernel_.setArg(kernel_arg++, cl_volume_dims);
        kernel_.setArg(kernel_arg++, (cl_float)voxel_size);
        kernel_.setArg(kernel_arg++, cl_model_pose);
        kernel_.setArg(kernel_arg++, cl_model_pose_inverse);
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
