#include "KernelComputeNormalVolumeWithWeights.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelComputeNormalVolumeWithWeights::kernel_name = "KernelComputeNormalVolumeWithWeights";

KernelComputeNormalVolumeWithWeights::KernelComputeNormalVolumeWithWeights(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelComputeNormalVolumeWithWeights::runKernel(
    VolumeBuffer & volume,
    VolumeBuffer & weights,
    VolumeBuffer & result
	)
{
	try {
        cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};

        //	void resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size);
        result.resize(volume.getVolumeCellCounts(), sizeof(float) * 4);
        result.setFloat4(Eigen::Array4f::Constant(0));

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume.getBuffer());
        kernel_.setArg(kernel_arg++, weights.getBuffer());
        kernel_.setArg(kernel_arg++, result.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);


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
