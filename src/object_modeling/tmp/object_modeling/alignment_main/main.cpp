#include <iostream>
using std::cout;
using std::endl;

#include "OpenCLNormals.h"

#include "alignment.h"

#include "frame_provider_base.h"
#include "frame_provider_png.h" // only this one for now...



int main(int argc, char* argv[])
{
	// load PNGs from a folder
	if (argc != 3) {
		cout << "Usage: EXE <path_to_cl_files> <path_to_png_files>" << endl;
		exit(1);
	}
	fs::path path_cl = argv[1];
	fs::path path_png = argv[2];
	boost::scoped_ptr<FrameProviderBase> frame_provider;
	//frame_provider.reset(new FrameProviderPNG(path));
	const float depth_factor = 10000.f; // my depth pngs are in 0.1mm units, so x10000 = meters.
	frame_provider.reset(new FrameProviderPNG(path_png, depth_factor));

	///////////////
	// start by initializing opencl
	boost::scoped_ptr<CL> cl_ptr;
	OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
	OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;
#if 0
	if (cl_intel_cpu) {
		platform_type = OPENCL_PLATFORM_INTEL;
		context_type = OPENCL_CONTEXT_CPU;
	}
	else if (cl_amd_cpu) {
		platform_type = OPENCL_PLATFORM_AMD;
		context_type = OPENCL_CONTEXT_CPU;
	}
#endif
	cl_ptr.reset(new CL(platform_type, context_type));
	if (!cl_ptr->isInitialized()) throw std::runtime_error ("Failed to initialize Open CL");

	//fs::path cl_path = "C:\\devlibs\\object_modeling\\OpenCLStaticLib";
	boost::shared_ptr<OpenCLAllKernels> all_kernels (new OpenCLAllKernels(*cl_ptr, path_cl));

	// needed to compute points and normals
	boost::shared_ptr<OpenCLNormals> opencl_normals(new OpenCLNormals(*cl_ptr, *all_kernels->getKernelsBuilderNormals()));

	//////////////
	// begin testing alignment
	ParamsAlignment params_alignment;
	// can change values here
	//params_alignment.icp_max_distance = 0.5; //etc
	Alignment alignment(all_kernels, params_alignment);

	const bool show_debug_images = true;
	alignment.setAlignDebugImages(show_debug_images); // slow but allows me to show debug images

	ParamsCamera params_camera;
	//params_camera.focal = Eigen::Array2f();
	//params_camera.center =  Eigen::Array2f();
	//params_camera.size =  Eigen::Array2i();
	//params_camera.min_max_depth =  Eigen::Array2f();

	cv::Mat depth, previous_depth;
	cv::Mat color, previous_color;
	while(true) {
		previous_depth = depth.clone();
		previous_color = color.clone();

		bool frame_valid = frame_provider->getNextFrame(color, depth);
		if (!frame_valid) break;

		if (previous_color.empty()) continue;

		// have previous and current

		// previous is target
		{ 
			ImageBuffer ib_target_color(*cl_ptr);
			ImageBuffer ib_target_depth(*cl_ptr);
			ImageBuffer ib_target_points(*cl_ptr);
			ImageBuffer ib_target_normals(*cl_ptr);
			ImageBuffer ib_target_weights(*cl_ptr);

			cv::Mat color_bgra;
			cv::cvtColor(previous_color, color_bgra, CV_BGR2BGRA);
			ib_target_color.writeFromBytePointer(color_bgra.data, color_bgra.total() * color_bgra.channels() * sizeof(uint8_t));
			ib_target_depth.writeFromBytePointer(previous_depth.data, previous_depth.total() * sizeof(float));

			// points and normals computed on GPU
			opencl_normals->depthImageToPoints(params_camera.size, params_camera.focal, params_camera.center, ib_target_depth, ib_target_points);
			const float normals_max_depth_sigmas = 3;
			const int normals_smooth_iterations = 2;
			opencl_normals->computeNormalsWithBuffers(params_camera.size.x(), params_camera.size.y(),
				ib_target_points, normals_max_depth_sigmas, normals_smooth_iterations, ib_target_normals);

			// trivial weights
			cv::Mat mat_weights = cv::Mat(color_bgra.size(), CV_32F, cv::Scalar(1));
			ib_target_weights.writeFromBytePointer(mat_weights.data, mat_weights.total() * sizeof(float));

			// "target" frame
			alignment.prepareFrame(ib_target_color, ib_target_points, ib_target_normals, ib_target_weights, params_camera);
		}


		// we will copy to GPU for both frames, though could also just assign ImageBuffers
		{
			ImageBuffer ib_move_color(*cl_ptr);
			ImageBuffer ib_move_depth(*cl_ptr);
			ImageBuffer ib_move_points(*cl_ptr);
			ImageBuffer ib_move_normals(*cl_ptr);
			ImageBuffer ib_move_mask(*cl_ptr); // assumed to be 32 bit its currently

			cv::Mat color_bgra;
			cv::cvtColor(color, color_bgra, CV_BGR2BGRA);
			ib_move_color.writeFromBytePointer(color_bgra.data, color_bgra.total() * color_bgra.channels() * sizeof(uint8_t));
			ib_move_depth.writeFromBytePointer(depth.data, depth.total() * sizeof(float));

			// points and normals computed on GPU
			opencl_normals->depthImageToPoints(params_camera.size, params_camera.focal, params_camera.center, ib_move_depth, ib_move_points);
			const float normals_max_depth_sigmas = 3;
			const int normals_smooth_iterations = 2;
			opencl_normals->computeNormalsWithBuffers(params_camera.size.x(), params_camera.size.y(),
				ib_move_points, normals_max_depth_sigmas, normals_smooth_iterations, ib_move_normals);

			// trivial mask (stupidly int 32 for now)
			cv::Mat depth_mask = depth > 0;
			cv::Mat depth_mask_int32;
			depth_mask.convertTo(depth_mask_int32, CV_32S);
			ib_move_mask.writeFromBytePointer(depth_mask_int32.data, depth_mask_int32.total() * sizeof(int));

			// and now do the alignment
			Eigen::Affine3f render_pose; // "base" pose
			render_pose = Eigen::Affine3f::Identity();
			Eigen::Affine3f initial_relative_pose;
			initial_relative_pose = Eigen::Affine3f::Identity();

			Eigen::Affine3f result_pose;
			int result_iterations;
			bool aligmnent_success = alignment.alignWithCombinedOptimization(ib_move_color, ib_move_points, ib_move_normals, ib_move_mask, params_camera, params_camera.getFullRenderRect(), render_pose, initial_relative_pose, result_pose, result_iterations);
			cout << "iterations: " << result_iterations << endl;

			cout << "aligmnent_success: " << aligmnent_success << endl;
			cout << "result_pose:\n" << result_pose.matrix() << endl;
		}

		if (show_debug_images) {
			std::vector<cv::Mat> debug_images;
			alignment.getAlignDebugImages(debug_images);
			const bool pause_after_each = false;
			for (int i = 0; i < debug_images.size(); ++i) {
				cv::imshow("debug_images", debug_images[i]);
				if (pause_after_each) cv::waitKey(0);
				else cv::waitKey(100);
			}
		}

		cout << "pausing after frame..." << endl;
		int k = cv::waitKey(0);
		if (k == 'q') break;
	}

	cout << "exiting normally..." << endl;
	return 0;
}
