#include "KernelHistogramSumCheckIndex.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelHistogramSumCheckIndex::kernel_name = "KernelHistogramSumCheckIndex";

KernelHistogramSumCheckIndex::KernelHistogramSumCheckIndex(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelHistogramSumCheckIndex::runKernel(
        VolumeBuffer & buffer_histogram_bin,
        VolumeBuffer & buffer_index,
        VolumeBuffer & buffer_sum,
        VolumeBuffer & buffer_count,
        int this_index,
        int index_range,
        float bin_min,
        float bin_max
        )
{
    try {
        cl_int4 cl_volume_dims = {buffer_histogram_bin.getVolumeCellCounts()[0], buffer_histogram_bin.getVolumeCellCounts()[1], buffer_histogram_bin.getVolumeCellCounts()[2], 0};

        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, buffer_histogram_bin.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_index.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_sum.getBuffer());
        kernel_.setArg(kernel_arg++, buffer_count.getBuffer());
        kernel_.setArg(kernel_arg++, this_index);
        kernel_.setArg(kernel_arg++, index_range);
        kernel_.setArg(kernel_arg++, bin_min);
        kernel_.setArg(kernel_arg++, bin_max);
        kernel_.setArg(kernel_arg++, cl_volume_dims);

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
