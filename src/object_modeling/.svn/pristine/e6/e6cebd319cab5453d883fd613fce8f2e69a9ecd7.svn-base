#include "KernelExtractVolumeFloat.h"

#include "BufferWrapper.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelExtractVolumeFloat::kernel_name = "KernelExtractVolumeFloat";

KernelExtractVolumeFloat::KernelExtractVolumeFloat(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelExtractVolumeFloat::runKernel(
	VolumeBuffer & volume,
	Eigen::Array3i const& voxel,
	float & result_float
	)
{
	try {
		cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};
		cl_int4 cl_voxel = {voxel[0], voxel[1], voxel[2], 0};

		BufferWrapper result_buffer_wrapper(cl_);
		result_buffer_wrapper.reallocate(1 * sizeof(float));


		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, volume.getBuffer());
		kernel_.setArg(kernel_arg++, result_buffer_wrapper.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, cl_voxel);


		// run kernel
		cl::NDRange global(1);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);

		// this is a bit silly
		std::vector<float> result_vector(1);
		result_buffer_wrapper.readToFloatVector(result_vector);
		result_float = result_vector[0];
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
