#include "KernelDotVolumeNormal.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelDotVolumeNormal::kernel_name = "KernelDotVolumeNormal";

KernelDotVolumeNormal::KernelDotVolumeNormal(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

#if 0
__kernel void KernelDotVolumeNormal(
    __global float *volume,
    __global float *result_dot,
    const int4 volume_dims,
    const float4 vector)
{
#endif

void KernelDotVolumeNormal::runKernel(
    VolumeBuffer & volume,
    VolumeBuffer & result,
    Eigen::Vector3f & vector
	)
{
	try {
        cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};
        cl_float4 cl_vector = {vector[0], vector[1], vector[2], 0};

        //	void resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size);
        result.resize(volume.getVolumeCellCounts(), sizeof(float));
        result.setFloat(0);

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, volume.getBuffer());
        kernel_.setArg(kernel_arg++, result.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);
        kernel_.setArg(kernel_arg++, cl_vector);


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
