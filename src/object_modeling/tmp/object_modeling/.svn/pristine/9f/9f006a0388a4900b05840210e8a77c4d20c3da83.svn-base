#include "stdafx.h"

#include "cll.h"
#include "util.h" // opencl util print
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using std::cout;
using std::endl;
#include <boost\filesystem.hpp>
namespace fs = boost::filesystem;
#include <Eigen\Geometry>

#include "OpenCLTSDF.h"
#include "OpenCLOptimize.h"

int main() {
	//OpenCLPlatformType opencl_platform_type = OPENCL_PLATFORM_NVIDIA;
	OpenCLPlatformType opencl_platform_type = OPENCL_PLATFORM_INTEL;
	OpenCLContextType opencl_context_type = OPENCL_CONTEXT_DEFAULT;
	bool opencl_debug = true;

	CL cl(opencl_platform_type, opencl_context_type);
	if (!cl.isInitialized()) {
		cout << "!cl.isInitialized()" << endl;
		exit(1);
	}

	// Keep this around, fool:
#if 0
	OpenCLTSDF opencl_tsdf(cl);
	opencl_tsdf.init(opencl_debug,
		525,525,
		320,240,
		640,480,
		64, 0.01f,
		0.03f, 0.03f,
		100,
		0.01f, 10.0f, 0.25f);


	// make up a fake frame
	//Eigen::Affine3f fake_pose = Eigen::Affine3f::Identity();
	//fake_pose.pretranslate(Eigen::Vector3f(0,0,0.5));
	Eigen::Affine3f fake_pose = Eigen::Affine3f(Eigen::Translation3f(0,0,0.5));
	const int rows = 480;
	const int cols = 640;
	const int image_size = rows * cols;
	std::vector<float> fake_object_depth_image(image_size, 0.3);
	std::vector<unsigned char> fake_color_image(image_size * opencl_tsdf.getColorChannelCount(), 0);
	opencl_tsdf.addFrame(fake_pose, fake_object_depth_image.data(), fake_color_image.data());

	std::vector<unsigned char> render_mask(image_size, 0);
	std::vector<unsigned char> render_colors(image_size * opencl_tsdf.getColorChannelCount(), 0);
	std::vector<float> render_points(image_size * 4, 0);
	std::vector<float> render_normals(image_size * 4, 0);
	opencl_tsdf.renderFrame(fake_pose, 0, 0, 640, 480, render_mask.data(), render_points.data(), render_normals.data(), render_colors.data());
#endif

#if 0
	OpenCLOptimize opencl_optimize(cl);
	opencl_optimize.init("", opencl_debug);
	
	Eigen::Affine3f fake_pose = Eigen::Affine3f(Eigen::Translation3f(0,0,0.5));
	const int rows = 480;
	const int cols = 640;
	const int image_size = rows * cols;
	int rr_x = 50, rr_y = 50, rr_w = 100, rr_h = 100;
	
	std::vector<float> frame_points(image_size * 4, 0);
	std::vector<float> frame_normals(image_size * 4, 0);
	std::vector<float> frame_intensity(image_size, 0);
	std::vector<float> frame_weights(image_size, 1.0);
	std::vector<float> frame_grad_x(image_size, 0);
	std::vector<float> frame_grad_y(image_size, 0);
	opencl_optimize.prepareFrameBuffers(0.05, cos(45 * 3.14159 / 180), 1.0, 1.0, 525,525,320,240, rr_x, rr_y, rr_w, rr_h, 640, 480, 1, frame_points.data(), frame_normals.data(), frame_intensity.data(), frame_weights.data(), frame_grad_x.data(), frame_grad_y.data());

	std::vector<float> render_points(image_size * 4, 0);
	std::vector<float> render_normals(image_size * 4, 0);
	std::vector<float> render_intensity(image_size, 0);
	std::vector<float> render_weights(image_size, 1.0);
	opencl_optimize.prepareRenderedAndErrorBuffers(525,525,320,240, rr_x, rr_y, rr_w, rr_h, render_points.data(), render_normals.data(), render_intensity.data(), render_weights.data());

	// test error
	std::vector<float> error_vector(opencl_optimize.getErrorVectorSize(), 0);
	opencl_optimize.errorICPAndColor(fake_pose, error_vector.data());

	// test df
	std::vector<float> error_matrix(opencl_optimize.getErrorMatrixSize(), 0);
	opencl_optimize.dfICPAndColor(fake_pose, error_matrix.data());
#endif


	////////////////
	// Test with vector add :
	// This probably doesn't work...for ref:
#if 0
	std::ifstream sourceFile("vector_add_kernel.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	cl.loadProgram(sourceCode);

	const int LIST_SIZE = 100;
	int *A = new int[LIST_SIZE]; 
	int *B = new int[LIST_SIZE];
	for(int i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}
	// Make kernel
	cl::Kernel kernel(cl.program, "vector_add");

	// Create memory buffers
	cl::Buffer bufferA (cl.context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
	cl::Buffer bufferB (cl.context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
	cl::Buffer bufferC (cl.context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int));

	// Copy lists A and B to the memory buffers
	cl.queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);
	cl.queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, LIST_SIZE * sizeof(int), B);

	// Set arguments to kernel
	kernel.setArg(0, bufferA);
	kernel.setArg(1, bufferB);
	kernel.setArg(2, bufferC);

	// Run the kernel on specific ND range
	cl::NDRange global(LIST_SIZE);
	cl::NDRange local(1);
	cl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

	// This is only necessary when CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	//cl.queue.enqueueBarrier(); // Peter added this

	// Read buffer C into a local list
	int *C = new int[LIST_SIZE];
	cl.queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, LIST_SIZE * sizeof(int), C);

	for(int i = 0; i < LIST_SIZE; i ++)
		std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl; 
#endif

	cout << "End of Console Test Reached" << endl;
	//getchar();
	return 0;
}