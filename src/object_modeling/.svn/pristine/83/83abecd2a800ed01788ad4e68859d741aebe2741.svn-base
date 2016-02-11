#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
using std::cout;
using std::endl;

#include "RenderBuffers.h"
#include "VolumeBuffer.h"

#include "KernelSetVolumeSDFBox.h"
#include "KernelRenderPointsAndNormals.h"
#include "KernelExtractVolumeSlice.h"
#include "KernelPointsToDepthImage.h"
#include "KernelRaytraceBox.h"
#include "KernelMinFloats.h"
#include "KernelRaytraceSpecial.h"

#include "params_camera.h"

#include "util.h"

#include "MeshUtilities.h"

// shouldn't really need this...debugging:
#include "volume_modeler_glfw.h"

int main(int argc, char* argv[])
{
	// don't use printf/scanf (and expect it to be in sync)
	std::ios_base::sync_with_stdio(false);

#ifdef _WIN32
#if 1
	// windows: buffer stdout better!
	const int console_buffer_size = 4096;
	char buf[console_buffer_size];
	setvbuf(stdout, buf, _IOLBF, console_buffer_size);
#else
	cout << "WARNING: SLOW COUT" << endl;
#endif
#endif

#ifdef _WIN32
	const fs::path cl_path_default = "C:\\devlibs\\object_modeling\\OpenCLStaticLib";
#else
	const fs::path cl_path_default = "/home/peter/checkout/object_modeling/OpenCLStaticLib";
#endif

	fs::path cl_path;
	fs::path output_folder = "output";
	bool pause = false;
	int frame_count = 36;
	float box_thickness = 0.2;
	bool raytrace_instead = false;
	int box_count = 1;
	float shape_rotation = 0; // degrees
	float noise_sigma = 0;
	bool special = false;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("cl_path", po::value<fs::path>(&cl_path), "cl_path")
		("output_folder", po::value<fs::path>(&output_folder), "output_folder")
		("pause", po::value<bool>(&pause)->zero_tokens(), "pause")
		("frame_count", po::value<int>(&frame_count), "frame_count")
		("box_thickness", po::value<float>(&box_thickness), "box_thickness")
		("raytrace_instead", po::value<bool>(&raytrace_instead)->zero_tokens(), "raytrace_instead")
		("box_count", po::value<int>(&box_count), "box_count")
		("shape_rotation", po::value<float>(&shape_rotation), "shape_rotation")
		("noise_sigma", po::value<float>(&noise_sigma), "noise_sigma")
		("special", po::value<bool>(&special)->zero_tokens(), "special")
		// more options
		;
	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
		po::notify(vm);
	}
	catch (std::exception & e) {
		cout << desc << endl;
		cout << e.what() << endl;
		exit(1);
	}
	if (vm.count("help")) {
		cout << "desc" << endl;
		exit(0);
	}
	if (cl_path.empty()) {
		cl_path = cl_path_default;
	}
	if (!fs::exists(output_folder) && !fs::create_directories(output_folder)) {
		cout << "Couldn't use or create output_folder: " << output_folder << endl;
		exit(1);
	}

	//////
	// for noise:
	boost::mt19937 rng; // default seed should be ok (and consistent)
	boost::normal_distribution<> nd(0.0, noise_sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);



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

	boost::shared_ptr<OpenCLAllKernels> all_kernels (new OpenCLAllKernels(*cl_ptr, cl_path));

	// also 3D debug
	// can just make null to disable
	boost::shared_ptr<VolumeModelerGLFW> glfw_ptr(new VolumeModelerGLFW(800,600));
	glfw_ptr->runInThread();

	// params...
	Eigen::Array3i cell_counts(256,256,256);
	float cell_size = 0.0025;
	Eigen::Array3f box_corner_from_origin(.1,.1,box_thickness/2);
	//Eigen::Affine3f box_pose = Eigen::Affine3f::Identity();
	ParamsCamera params_camera;
	params_camera.size = Eigen::Array2i(640,480);
	params_camera.focal = Eigen::Array2f(525,525);
	params_camera.setCenterFromSize();


	VolumeBuffer volume(all_kernels, cell_counts, sizeof(float));
	volume.setFloat(-1);
	VolumeBuffer weights(all_kernels, cell_counts, sizeof(float));
	weights.setFloat(1); // all valid!

	// the volume
	Eigen::Affine3f volume_pose = Eigen::Affine3f::Identity();
	{
		Eigen::Vector3f t(-cell_counts.cast<float>() / 2 * cell_size);
		volume_pose.pretranslate(t);
	}

	if (glfw_ptr) {
		MeshVertexVectorPtr vertices(new MeshVertexVector);
		getBoundingLines(cell_counts, cell_size, volume_pose, Eigen::Array4ub(255,255,255,255), *vertices);
		glfw_ptr->updateLines("volume_lines", vertices);
	}

	// put all the box poses in a list
	typedef std::vector<boost::shared_ptr<Eigen::Affine3f> > PosePtrList;
	PosePtrList box_pose_list;
	for (int box = 0; box < box_count; ++box) {
		Eigen::Affine3f box_pose = Eigen::Affine3f::Identity();
		box_pose.rotate(Eigen::AngleAxisf( shape_rotation * M_PI / 180.f, Eigen::Vector3f(0,-1,0)));
		box_pose.rotate(Eigen::AngleAxisf( ((float)box / (float)box_count) * M_PI, Eigen::Vector3f(0,-1,0)));
		box_pose_list.push_back(PosePtrList::value_type(new Eigen::Affine3f(box_pose)));
	}

	if (special) {
		// todo: set up special volume
	}
	else {
		// put them all in the volume (awkwardly using auxilary volumes)
		KernelSetVolumeSDFBox _KernelSetVolumeSDFBox(*all_kernels);
		KernelMinFloats _KernelMinFloats(*all_kernels);
		VolumeBuffer temp_sdf_buffer(all_kernels, volume.getVolumeCellCounts(), sizeof(float));
		VolumeBuffer temp_index_buffer(all_kernels, volume.getVolumeCellCounts(), sizeof(int));
		temp_index_buffer.setInt(-1);
		for (int box = 0; box < box_pose_list.size(); ++box) {
			Eigen::Affine3f box_pose = *box_pose_list[box];
			//temp_sdf_buffer.setFloat(-1); // no need to reset...sets all in kernel
			_KernelSetVolumeSDFBox.runKernel(
				temp_sdf_buffer,
				cell_size,
				volume_pose,
				box_corner_from_origin,
				box_pose,
				params_camera.size,
				params_camera.focal,
				params_camera.center
				);

			// min with running buffer
			_KernelMinFloats.runKernel(volume.getBuffer(), temp_index_buffer.getBuffer(), temp_sdf_buffer.getBuffer(), box, volume.getSizeInCells());
		}
	}

	RenderBuffers render_buffers(all_kernels);
	render_buffers.setSize(params_camera.size[0], params_camera.size[1]);

	PosePtrList camera_list;

	for (int frame = 0; frame < frame_count; ++frame) {
		cout << frame << endl;

		Eigen::Affine3f camera_pose = Eigen::Affine3f::Identity();
		camera_pose = Eigen::AngleAxisf(frame * 2 * M_PI / frame_count, Eigen::Vector3f(0,-1,0));
		camera_pose.translate(Eigen::Vector3f(0,0,-1));

		camera_list.push_back(PosePtr(new Eigen::Affine3f(camera_pose)));

		if (glfw_ptr) {
			UpdateInterface::PoseListPtrT cameras(new UpdateInterface::PoseListT);
			cameras->push_back(camera_pose);
			glfw_ptr->updateCameraList("cameras", cameras);
		}

		const int mask_value = 1; // 0 seems fine..., 1 better for debugging

		cv::Mat color, normals;
		render_buffers.resetAllBuffers();

		if (raytrace_instead) {
			if (special) {
				Eigen::Affine3f object_pose = Eigen::Affine3f::Identity();

				KernelRaytraceSpecial _KernelRaytraceSpecial(*all_kernels);
				_KernelRaytraceSpecial.runKernel(
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					camera_pose.inverse(), // volume pose irrelevant
					object_pose,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
			}
			else {
				KernelRaytraceBox _KernelRaytraceBox(*all_kernels);
				for (int box = 0; box < box_pose_list.size(); ++box) {
					Eigen::Affine3f box_pose = *box_pose_list[box];
					_KernelRaytraceBox.runKernel(
						render_buffers.getImageBufferMask(),
						render_buffers.getImageBufferPoints(),
						render_buffers.getImageBufferNormals(),
						camera_pose.inverse(), // volume pose irrelevant
						box_corner_from_origin,
						box_pose,
						params_camera.focal,
						params_camera.center,
						params_camera.min_max_depth[0],
						params_camera.min_max_depth[1],
						mask_value);
				}
			}
		}
		else {
			KernelRenderPointsAndNormals _KernelRenderPointsAndNormals(*all_kernels);
			_KernelRenderPointsAndNormals.runKernel(
				volume,
				weights,
				render_buffers.getImageBufferMask(),
				render_buffers.getImageBufferPoints(),
				render_buffers.getImageBufferNormals(),
				cell_size,
				camera_pose.inverse() * volume_pose,
				params_camera.focal,
				params_camera.center,
				params_camera.min_max_depth[0],
				params_camera.min_max_depth[1],
				mask_value);
		}

		render_buffers.getRenderPretty(color, normals);
		cv::imshow("normals", normals);

#if 0
		// also look at a slice...
		KernelExtractVolumeSlice _KernelExtractVolumeSlice(*all_kernels);
		ImageBuffer slice(all_kernels->getCL());
		_KernelExtractVolumeSlice.runKernel(
			volume,
			1,
			cell_counts[1] / 2,
			slice
			);
		cv::Mat slice_mat = slice.getMat();
		double min,max;
		cv::minMaxLoc(slice_mat,&min,&max);
		//cout << "min max: " << min << " " << max << endl;
		slice_mat -= min;
		cv::Mat slice_mat_display;
		slice_mat.convertTo(slice_mat_display, CV_8U, 255/max);
		cv::imshow("slice_mat_display", slice_mat_display);
#endif

		// mask to look at
#if 1
		cv::Mat render_mask_mat = render_buffers.getImageBufferMask().getMat();
		cv::imshow("render_mask_mat", render_mask_mat * 255 * 255);
#endif

		// look at points in the volume
		if (glfw_ptr) {
			Eigen::Matrix4Xf points = render_buffers.getImageBufferPoints().getMatrix4Xf();
			points = camera_pose * points;
			MeshVertexVectorPtr vertices(new MeshVertexVector);
			for (int col = 0; col < points.cols(); ++col) {
				MeshVertex v;
				v.p = points.col(col);
				v.c = Eigen::Array4ub::Constant(255);
				vertices->push_back(v);
			}
			glfw_ptr->updatePointCloud("vertices", vertices);
		}


		// get depth image from points and save
		cv::Mat depth_to_save;
		{
			KernelPointsToDepthImage _KernelPointsToDepthImage(*all_kernels);
			ImageBuffer depth_buffer(all_kernels->getCL());
			_KernelPointsToDepthImage.runKernel(render_buffers.getImageBufferPoints(), depth_buffer);
			cv::Mat depth_mat = depth_buffer.getMat();

			if (noise_sigma > 0) {
				cv::MatIterator_<float> mat_iter = depth_mat.begin<float>();
				for( ; mat_iter != depth_mat.end<float>(); ++mat_iter) {
					if (*mat_iter > 0) *mat_iter += var_nor();
				}
			}

			const float depth_factor = 10000.f;
			cv::Mat depth_mat_16;
			depth_mat.convertTo(depth_mat_16, CV_16U, depth_factor);
			depth_to_save = depth_mat_16;
		}

		// save whatever we should 
		{
			fs::path filename = (boost::format("depth_%05d.png") % frame).str();
			cv::imwrite( (output_folder / filename).string(), depth_to_save);
		}

		// look at images
		if (pause) {
			cout << "pause..." << endl;
			cv::waitKey(0);
		}
		else {
			cv::waitKey(1);
		}

	} // for loop


	// now save cameras
	{
		fs::path filename = "camera_poses.txt";
		std::ofstream ofs ( (output_folder / filename).string().c_str() );
		if (!ofs.good()) {
			cout << "failed to write camera poses" << endl;
			exit(1);
		}
		for (size_t i = 0; i < camera_list.size(); ++i) {
			ofs << EigenUtilities::transformToString(*camera_list[i]) << endl;
		}
		cout << "saved camera poses to: " << filename << endl;
	}

	// also generate and save reference mesh!!
	{
		Eigen::Array4ub color = Eigen::Array4ub(0,255,0,255);
		MeshPtr mesh = MeshUtilities::generateMesh(volume, weights, cell_size, color, volume_pose);
		fs::path filename = "mesh_volume.ply";
		MeshUtilities::saveMesh(mesh->vertices, mesh->triangles, output_folder / filename);
		cout << "saved mesh to: " << filename << endl;
	}

	// write mesh directly instead of from volume as well...
	{
		Eigen::Array4ub color = Eigen::Array4ub(0,255,0,255);
		MeshPtr mesh(new Mesh);
		for (int box = 0; box < box_pose_list.size(); ++box) {
			Eigen::Affine3f box_pose = *box_pose_list[box];
			Mesh temp_mesh;
			std::vector<Eigen::Vector3f> corners = getBoxCorners(-box_corner_from_origin, box_corner_from_origin, box_pose);
			getMeshForBoxCorners(corners, color, temp_mesh.vertices, temp_mesh.triangles);
			MeshUtilities::appendMesh(mesh->vertices, mesh->triangles, temp_mesh.vertices, temp_mesh.triangles);
		}
		fs::path filename = "mesh_box_corners.ply";
		MeshUtilities::saveMesh(mesh->vertices, mesh->triangles, output_folder / filename);
		cout << "saved mesh to: " << filename << endl;
	}


	if (glfw_ptr) {
		glfw_ptr->destroy();
		glfw_ptr->join();
	}

	return 0;
}
