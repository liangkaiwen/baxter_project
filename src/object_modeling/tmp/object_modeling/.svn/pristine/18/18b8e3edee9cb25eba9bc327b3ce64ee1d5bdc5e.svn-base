#pragma once

#include "cll.h"
#include "OpenCLAllKernels.h"
#include "ImageBuffer.h"

// todo: rename this to ...Points as it deals with more than just normals
// actual todo: eliminate this file entirely and use new kernel interface instead...
class OpenCLNormals
{
private:
	OpenCLNormals(const OpenCLNormals& other);
	void operator=(const OpenCLNormals& other);

public:
	static const int local_work_size_side = 16;// public for debugging?

	OpenCLNormals(boost::shared_ptr<OpenCLAllKernels> all_kernels);

	void computeNormalsWithBuffers(
		const ImageBuffer & image_buffer_points,
		float max_sigmas,
		int smooth_iterations,
		ImageBuffer & image_buffer_normals);

protected:
	//////////
	// members
	boost::shared_ptr<OpenCLAllKernels> all_kernels_;

	cl::Kernel compute_normals_kernel;
	cl::Kernel smooth_normals_kernel;

	ImageBuffer image_buffer_normals_temp_1;
	ImageBuffer image_buffer_normals_temp_2;
};

