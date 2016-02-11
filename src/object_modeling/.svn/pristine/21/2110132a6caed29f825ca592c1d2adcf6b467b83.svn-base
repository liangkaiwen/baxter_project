#include "KernelBetterNormal.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelBetterNormal::kernel_name = "KernelBetterNormal";

KernelBetterNormal::KernelBetterNormal(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelBetterNormal::runKernel(
            VolumeBuffer & volume_1,
            VolumeBuffer & volume_counts_1,
            VolumeBuffer & volume_normals_1,
            VolumeBuffer & volume_2,
            VolumeBuffer & volume_counts_2,
            VolumeBuffer & volume_normals_2,
            VolumeBuffer & result_volume,
            VolumeBuffer & result_counts,
            Eigen::Affine3f const& pose,
			float minimum_relative_count
            )
{
	try {
        cl_int4 cl_volume_dims = {volume_1.getVolumeCellCounts()[0], volume_1.getVolumeCellCounts()[1], volume_1.getVolumeCellCounts()[2], 0};

        cl_float16 cl_pose = getCLPose(pose);

        //	void resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size);
        result_volume.resize(volume_1.getVolumeCellCounts(), sizeof(float));
        result_counts.resize(volume_1.getVolumeCellCounts(), sizeof(float));

		// assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_counts_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_normals_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_2.getBuffer());
        kernel_.setArg(kernel_arg++, volume_counts_2.getBuffer());
        kernel_.setArg(kernel_arg++, volume_normals_2.getBuffer());
        kernel_.setArg(kernel_arg++, result_volume.getBuffer());
        kernel_.setArg(kernel_arg++, result_counts.getBuffer());
        kernel_.setArg(kernel_arg++, cl_volume_dims);
        kernel_.setArg(kernel_arg++, cl_pose);
        kernel_.setArg(kernel_arg++, minimum_relative_count);

		// run kernel
		cl::NDRange global(cl_volume_dims.s[0], cl_volume_dims.s[1], cl_volume_dims.s[2]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
