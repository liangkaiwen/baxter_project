#include "KernelHistogramMaxCheckIndex.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelHistogramMaxCheckIndex::kernel_name = "KernelHistogramMaxCheckIndex";

KernelHistogramMaxCheckIndex::KernelHistogramMaxCheckIndex(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelHistogramMaxCheckIndex::runKernel(
        VolumeBuffer & buffer_histogram_bin,
        VolumeBuffer & buffer_previous_index,
        VolumeBuffer & buffer_index,
        VolumeBuffer & buffer_value,
        int index,
        int index_range
		)
{
	try {
		cl_int4 cl_volume_dims = {buffer_histogram_bin.getVolumeCellCounts()[0], buffer_histogram_bin.getVolumeCellCounts()[1], buffer_histogram_bin.getVolumeCellCounts()[2], 0};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer_histogram_bin.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_previous_index.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_index.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_value.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);
        kernel_.setArg(kernel_arg++, (cl_int) index);
        kernel_.setArg(kernel_arg++, (cl_int) index_range);

		// run kernel
		cl::NDRange global(buffer_histogram_bin.getVolumeCellCounts()[0], buffer_histogram_bin.getVolumeCellCounts()[1], buffer_histogram_bin.getVolumeCellCounts()[2]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
