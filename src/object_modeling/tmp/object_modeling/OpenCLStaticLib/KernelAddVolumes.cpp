#include "KernelAddVolumes.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelAddVolumes::kernel_name = "KernelAddVolumes";

KernelAddVolumes::KernelAddVolumes(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


#if 0
__kernel void KernelAddVolumes(
    __global float *volume_1,
    __global float *volume_counts_1,
    __global float *volume_2,
    __global float *volume_counts_2,
    __global float *result_volume,
    __global float *result_counts,
    const int4 volume_dims,
#endif

void KernelAddVolumes::runKernel(
            VolumeBuffer & volume_1,
            VolumeBuffer & volume_counts_1,
            VolumeBuffer & volume_2,
            VolumeBuffer & volume_counts_2,
            VolumeBuffer & result_volume,
            VolumeBuffer & result_counts
            )
{
	try {
        cl_int4 cl_volume_dims = {volume_1.getVolumeCellCounts()[0], volume_1.getVolumeCellCounts()[1], volume_1.getVolumeCellCounts()[2], 0};

        //	void resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size);
        result_volume.resize(volume_1.getVolumeCellCounts(), sizeof(float));
        result_counts.resize(volume_1.getVolumeCellCounts(), sizeof(float));

		// assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_counts_1.getBuffer());
        kernel_.setArg(kernel_arg++, volume_2.getBuffer());
        kernel_.setArg(kernel_arg++, volume_counts_2.getBuffer());
        kernel_.setArg(kernel_arg++, result_volume.getBuffer());
        kernel_.setArg(kernel_arg++, result_counts.getBuffer());
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
