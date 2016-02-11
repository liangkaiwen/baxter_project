#include "stdafx.h"
#include "ObjectModeler.h"

// pcl
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/warp_point_rigid_6d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "opencvUtil.h"
#include "pclUtil.hpp"

// g2o lib
#include <sac_model_registration_reprojection.h>

// opencl lib
#include <OpenCLOptimize.h>
#include <Noise.h>

// eigen utilities lib
#include <EigenUtilities.h>


using std::cout;
using std::endl;

ObjectModeler::ObjectModeler(Parameters& params)
	:	params(params),
	mask_object(params),
	feature_type(FEATURE_TYPE_FAST),
	light_direction(-1,1,1),
	g2o_stereo_projector(
		Eigen::Vector2f(params.camera_focal_x, params.camera_focal_y), 
		Eigen::Vector2f(params.camera_center_x, params.camera_center_y),
		params.camera_stereo_baseline),
	learn_histogram(params),
	tc_masked_points("masked_points", true),
	tc_model_kp("model_kp", false),
	tls_volume_corners("volume_corners", true),
	tls_patch_volume_debug("tls_patch_volume_debug", true),
	tm_patch_volume_debug("tm_patch_volume_debug", true),
	tls_patch_volume_normals("tls_patch_volume_normals", true),
	tls_graph_edges("tls_graph_edges", true),
	tls_frustum("tls_frustum", true),
	render_on_next_viewer_update(false),
	generate_mesh_on_next_viewer_update(false),
	input_frame_counter(0),
	output_frame_counter(0),
	previous_relative_transform(Eigen::Affine3f::Identity()),
	pv_test_image_index(0),
	is_active(false),
	is_tracking(false),
	is_stopping(false),
	previous_error_color(0),
	initialize_learn_histogram_next_frame(false),
	learning_histogram_hand(false),
	learning_histogram_object(false),
	paused_by_user(false),
	any_failures_save_exception(false),
	save_screenshot_on_next_viewer_update(false),
	save_screenshot_counter(0),
	loop_closure_count(0)
{
	object_pose.setIdentity();

	if (!params.disable_cloud_viewer) {
		cloud_viewer_ptr.reset(new pcl::visualization::CloudViewer("Cloud Viewer"));
		cloud_viewer_ptr->runOnVisualizationThreadOnce (boost::bind( &ObjectModeler::cloudViewerOnVisualizationThreadOnce, boost::ref(*this), _1));
		cloud_viewer_ptr->runOnVisualizationThread (boost::bind( &ObjectModeler::cloudViewerOnVisualizationThread, boost::ref(*this), _1));
		cloud_viewer_ptr->registerKeyboardCallback<ObjectModeler> (&ObjectModeler::cloudViewerKeyboardCallback, boost::ref(*this), NULL);
	}

	// init opencl
	OpenCLPlatformType platform_type = params.opencl_nvidia ? OPENCL_PLATFORM_NVIDIA : OPENCL_PLATFORM_INTEL;
	OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;
	if (params.opencl_context_cpu) context_type = OPENCL_CONTEXT_CPU;
	else if (params.opencl_context_gpu) context_type = OPENCL_CONTEXT_GPU;

	cl_ptr.reset(new CL(platform_type, context_type));
	if (!cl_ptr->isInitialized()) throw std::runtime_error ("Failed to initialize Open CL");

	all_kernels_ptr.reset(new OpenCLAllKernels(*cl_ptr, params.opencl_cl_path, params.opencl_debug, params.opencl_fast_math));

	// TSDF kernels:
	if (params.use_patch_volumes) {
		// trivial "0" patch volume
		pv_list.push_back(ObjectModeler::PVStructPtr(new PVStruct()));
	}
	else {
		opencl_tsdf_ptr.reset(new OpenCLTSDF(*cl_ptr, *all_kernels_ptr->getKernelsBuilderTSDF(),
			params.camera_focal_x, params.camera_focal_y, params.camera_center_x, params.camera_center_y, params.camera_size_x, params.camera_size_y,
			params.volume_cell_count_x, params.volume_cell_count_y, params.volume_cell_count_z, params.volume_cell_size,
			0,0,0,
			params.volume_max_weight));
	}


	// Optimize kernels
	opencl_optimize_ptr.reset(new OpenCLOptimize(*cl_ptr, *all_kernels_ptr->getKernelsBuilderOptimize(), getImageChannelCount()));

	// normals kernels
	opencl_normals_ptr.reset(new OpenCLNormals(*cl_ptr, *all_kernels_ptr->getKernelsBuilderNormals()));

	// images kernels
	opencl_images_ptr.reset(new OpenCLImages(*cl_ptr, *all_kernels_ptr->getKernelsBuilderImages()));


	// init histograms
	if (params.mask_hand) {
		if (!loadBothHistograms()) {
			// we have tried and failed to load histograms
			if (!isInLiveMode() && !params.mask_hand_hist_learn && !params.mask_object_hist_learn) throw std::runtime_error ("Failed to load histograms (required at launch for offline mode unless learning histograms offline)");
		}
	}

	// initialize ofs
	if (params.save_poses) {
		fs::path dump_path = prepareDumpPath();
		fs::path object_poses_filename = dump_path / "object_poses_sequential.txt";
		fs::path camera_poses_filename = dump_path / "camera_poses_sequential.txt";
		ofs_object_pose.open(object_poses_filename.string().c_str());
		ofs_camera_pose.open(camera_poses_filename.string().c_str());
	}

	if (params.load_object_poses) {
		// if absolute path, don't use folder_input
		fs::path full_path_for_object_poses;
		if (params.file_input_object_poses.is_absolute()) {
			full_path_for_object_poses = params.file_input_object_poses;
		}
		else {
			full_path_for_object_poses = params.folder_input / params.file_input_object_poses;
		}
		if (!readTransforms(full_path_for_object_poses, input_object_poses)) {
			cout << "Bad full_path_for_object_poses: " << full_path_for_object_poses << endl;
			cout << "params.file_input_object_poses: " << params.file_input_object_poses << endl;
			cout << "params.file_input_object_poses.root_name(): " << params.file_input_object_poses.root_name() << endl;
			cout << "params.file_input_object_poses.root_directory(): " << params.file_input_object_poses.root_directory() << endl;

			throw std::runtime_error ("Bad file_input_object_poses");
		}
	}
	if (params.debug_do_load_pv_info) {
		if (!readTransforms(params.folder_input / params.file_input_loop_poses, input_loop_poses)) throw std::runtime_error ("!readTransforms(params.folder_input / params.file_input_loop_poses, input_loop_poses)");
		if (!readWhichPVs(params.folder_input / params.file_input_loop_which_pvs, input_loop_which_pvs)) throw std::runtime_error ("!readWhichPVs(params.folder_input / params.file_input_loop_which_pvs, input_loop_which_pvs)");
		if (input_loop_which_pvs.size() != input_loop_poses.size()) throw std::runtime_error ("input_loop_which_pvs.size() != input_loop_poses.size()");
	}

	// init frustum
	Eigen::Vector2f resolution(params.camera_size_x, params.camera_size_y);
	Eigen::Vector2f proj_f(params.camera_focal_x, params.camera_focal_y);
	Eigen::Vector2f proj_c(params.camera_center_x, params.camera_center_y);
	Frustum frustum(Eigen::Affine3f::Identity(), resolution, proj_f, proj_c, params.camera_z_min, params.camera_z_max);
	tls_frustum.setLineSet(getLineSetEdges(frustum.getPoints()));

	// save command line
	fs::path dump_path = prepareDumpPath();
	fs::path dump_command_line = dump_path / "command_line.txt";
	ofstream ofs(dump_command_line.string().c_str());
	ofs << params.full_command_line << endl;
	ofs.close();

	// avoid some weird coloring issues?
	// initialize color map?
	segment_color_map[0] = cv::Vec3b(255,255,255);
	prepareSegmentColorMap(1000);
	cout << "prepareSegmentColorMap(1000)" << endl;
}

bool ObjectModeler::readTransforms(fs::path const& file, std::vector<std::pair<bool, Eigen::Affine3f> >& vector)
{
	vector.clear();
	if (!fs::exists(file) || !fs::is_regular_file(file)) {
		cout << "bad transforms file: " << file.string() << endl;
		return false;
	}
	std::ifstream ifs(file.string().c_str());
	while (ifs.good()) {
		std::string line;
		std::getline(ifs, line);
		if (!line.empty()) {
			Eigen::Affine3f t = EigenUtilities::stringToTransform(line);
			vector.push_back(std::make_pair(true, t));
		}
		else {
			vector.push_back(std::make_pair(false, Eigen::Affine3f::Identity()));
		}
	}
	return true;
}

bool ObjectModeler::readWhichPVs(fs::path const& file, std::vector<std::vector<int> >& vector)
{
	vector.clear();
	if (!fs::exists(file) || !fs::is_regular_file(file)) {
		cout << "bad which pvs file: " << file.string() << endl;
		return false;
	}
	std::ifstream ifs(file.string().c_str());
	while (ifs.good()) {
		std::string line;
		std::getline(ifs, line);
		std::istringstream instr(line);
		vector.push_back(std::vector<int>());
		std::copy(std::istream_iterator<int>(instr), std::istream_iterator<int>(), std::back_inserter<std::vector<int> >(vector.back()));
	}
	return true;
}

void ObjectModeler::setActive(bool active)
{
	is_active = active;
	if (is_active) previous_relative_transform.setIdentity();
}

bool ObjectModeler::isEmpty() const
{
	if (params.use_patch_volumes) {
		return pv_list.size() <= 1;
	}
	else {
		return opencl_tsdf_ptr->isEmpty();
	}
}

void ObjectModeler::stop()
{
	// mutex?
	is_stopping = true;
}

bool ObjectModeler::wasStopped()
{
	return cloud_viewer_ptr && cloud_viewer_ptr->wasStopped();
}

bool ObjectModeler::isInLiveMode()
{
	// the current proxy for live mode:
	return params.folder_input.empty();
}

void ObjectModeler::cloudViewerOnVisualizationThreadOnce (pcl::visualization::PCLVisualizer& viewer)
{
	Eigen::Vector3f pos(0,0,-1);
	Eigen::Vector3f look(0,0,1);
	Eigen::Vector3f up(0,-1,0);
	viewer.setCameraPose(pos[0], pos[1], pos[2], look[0], look[1], look[2], up[0], up[1], up[2]);
	if (!params.show_axes) viewer.removeCoordinateSystem(); // remove axes?
	if (params.white_background) viewer.setBackgroundColor(1, 1, 1); // white background is prettier
}

void ObjectModeler::cloudViewerOnVisualizationThread (pcl::visualization::PCLVisualizer& viewer)
{
	if (is_stopping) {
		viewer.close();
		return;
	}

	// gotta update all
	tc_masked_points.update(viewer);
	tc_model_kp.update(viewer);
	tls_volume_corners.update(viewer);
	tls_patch_volume_debug.update(viewer);
	tm_patch_volume_debug.update(viewer);
	tls_patch_volume_normals.update(viewer);
	tls_graph_edges.update(viewer);
	tls_frustum.update(viewer);
	{
		boost::mutex::scoped_lock lock(mutex_pv_meshes_show);
		for (int i = 0; i < tm_pv_show_generated_list.size(); ++i) {
			tm_pv_show_generated_list[i]->update(viewer);
		}
		for (int i = 0; i < tc_pv_compare_to_mesh_list.size(); ++i) {
			tc_pv_compare_to_mesh_list[i]->update(viewer);
		}
	}

	if (save_screenshot_on_next_viewer_update) {
		viewer.saveScreenshot(getPNGFilenameWithIndex("viewer-screenshot", save_screenshot_counter).string());
		save_screenshot_on_next_viewer_update = false;
	}

	if (render_on_next_viewer_update) {
		boost::mutex::scoped_lock lock(mutex_volumes);

		render_on_next_viewer_update = false;

		std::vector<pcl::visualization::Camera> cameras;
		viewer.getCameras(cameras);
		if (cameras.empty()) throw std::runtime_error ("cameras.empty()");
		Eigen::Matrix4d view_mat;
		cameras[0].computeViewMatrix(view_mat);
		Eigen::Affine3f camera_pose(view_mat.cast<float>());
		// camera appears to have negative y, negative z
		// indeed, camera is using opengl convention of x-right, y-up, z-back
		Eigen::Affine3f camera_pose_corrector(Eigen::AngleAxisf(M_PI, Eigen::Vector3f(1,0,0)));
		Eigen::Affine3f custom_object_pose = camera_pose_corrector * camera_pose * object_pose;
		// transform the light to keep consistent relative to camera?
		Eigen::Vector3f light_direction_transformed = (camera_pose_corrector * camera_pose).rotation() * light_direction;

		// patch volume image
		if (params.use_patch_volumes) {
			cv::Mat pv_rendered_colors;
			cv::Mat pv_rendered_normals;
			cv::Mat pv_rendered_depth;
			cv::Mat pv_rendered_segments;
			renderPatchVolumesWithPose(custom_object_pose, light_direction_transformed, params.render_viewer_scale, pv_rendered_colors, pv_rendered_normals, pv_rendered_depth, pv_rendered_segments);

			cv::Mat pv_pair = create1x2(pv_rendered_colors, pv_rendered_normals);
			enqueueShowImage("pv_render_once", pv_pair);
			enqueueShowImage("pv_render_once_depth", pv_rendered_depth);

			cv::Mat pv_rendered_segments_c = randomlyColorSegments(pv_rendered_segments, cv::Mat());
			enqueueShowImage("pv_render_once_segments", pv_rendered_segments_c);
		}
		else {
			cv::Mat rendered_colors;
			cv::Mat rendered_normals;
			cv::Mat rendered_depth;
			renderVolumeWithPose(custom_object_pose, light_direction_transformed, params.render_viewer_scale, rendered_colors, rendered_normals, rendered_depth);

			enqueueShowImage("render_once", create1x2(rendered_colors, rendered_normals));
			enqueueShowImage("render_once_depth", rendered_depth);
		}
	}

	if (generate_mesh_on_next_viewer_update) {
		generate_mesh_on_next_viewer_update = false;
		// This locks mutex_volumes
		generateMesh();
	}
}

void ObjectModeler::processKey(char key)
{
	// both render thread and main thread can enter this...
	boost::mutex::scoped_lock lock(mutex_process_key);
	// note that the toggleState() calls are also each separately synchronized with the respective toggle object

	// From PCL viewer, can't use: p: points, w: wireframe, s: surface, j, f, e, q, g, u, r, h
	// OH, note that you might be able to use CAPITAL LETTERS.  Yell your commands!

	// 'c' is just display camera params
	if (key == 'c') {
		// a bit hacky to change param value, and this cause problems when saving an error movie
		params.combined_debug_images = !params.combined_debug_images;
	}
	else if (key == 'a') {
		if (isInLiveMode()) {
			if (!is_active && !isReadyToActivate()) return;
			setActive(!is_active);
		}
	}
	else if (key == 'o') {
		if (isInLiveMode()) {
			if (!learning_histogram_hand && !learning_histogram_object) {
				learning_histogram_object = true;
				initialize_learn_histogram_next_frame = true;
				setActive(true);
			}
		}
	}
	else if (key == 'i') {
		if (isInLiveMode()) {
			if (!learning_histogram_hand && !learning_histogram_object) {
				learning_histogram_hand = true;
				initialize_learn_histogram_next_frame = true;
				setActive(true);
			}
		}
	}
	else if (key == 't') {
		tc_masked_points.toggleState();
	}
	else if (key == 'l') {
		// free(ish)
		params.pv_color_all_boxes = !params.pv_color_all_boxes;
		updatePatchVolumeVisualizer(); // is this safe?
	}
	else if (key == 'k') {
		tc_model_kp.toggleState();
	}
	else if (key == 's') {
		// single volume
		tls_volume_corners.toggleState();
		// patch volumes
		tls_patch_volume_debug.toggleState();
		// patch volumes as mesh
		tm_patch_volume_debug.toggleState();
		// normals
		tls_patch_volume_normals.toggleState();
	}
	else if (key == 'd') {
		tls_graph_edges.toggleState();
	}
	else if (key == 'n') {
		// temporary?
		updatePatchVolumeVisualizer();
	}
	else if (key == 'm') {
		generate_mesh_on_next_viewer_update = true;
	}
	else if (key == ',') {
		// debug pvs as points
		boost::mutex::scoped_lock lock(mutex_pv_meshes_show);
		for (int i = 0; i < tc_pv_compare_to_mesh_list.size(); ++i) {
			tc_pv_compare_to_mesh_list[i]->toggleState();
		}
	}
	else if (key == 'z') {
		// free
	}
	else if (key == 'x') {
		tls_frustum.toggleState();
	}
	else if (key == 'v') {
		render_on_next_viewer_update = true;
	}
	else if (key == 'y') {
		// debug pvs as mesh
		boost::mutex::scoped_lock lock(mutex_pv_meshes_show);
		for (int i = 0; i < tm_pv_show_generated_list.size(); ++i) {
			tm_pv_show_generated_list[i]->toggleState();
		}
	}
	else if (key == 'b') {
		paused_by_user = !paused_by_user;
	}
}

void ObjectModeler::cloudViewerKeyboardCallback (const pcl::visualization::KeyboardEvent& keyboard_event, void* cookie)
{
	if (keyboard_event.keyDown()) {
		processKey(keyboard_event.getKeyCode());
	}
}

bool ObjectModeler::isReadyToActivate()
{
	bool result = true;
	if (params.mask_hand) {

		if (!mask_object.histogram_hand.data) {
			cout << "No hand histogram loaded" << endl;
			result = false;
		}
		if (!mask_object.histogram_object.data) {
			cout << "No object histogram loaded" << endl;
			result = false;
		}
	}
	return result;
}

fs::path ObjectModeler::prepareDumpPath(fs::path subfolder)
{
	fs::path dump_path = params.folder_output;
	if (dump_path.empty()) dump_path = "dump";
	if (!subfolder.empty()) dump_path = dump_path / subfolder;
	if (!fs::exists(dump_path)) fs::create_directories(dump_path);
	return dump_path;
}

fs::path ObjectModeler::prepareDumpPath()
{
	return prepareDumpPath("");
}

void ObjectModeler::savePNGWithIndex(const std::string& filename_prefix, const cv::Mat& image, int index)
{
	cv::imwrite(getPNGFilenameWithIndex(filename_prefix, index).string(), image);
}

fs::path ObjectModeler::getPNGFilenameWithIndex(const std::string& filename_prefix, int index)
{
	fs::path dump_path = prepareDumpPath();
	fs::path dump_filename = dump_path / (filename_prefix + (boost::format("-%05d.png") % index).str());
	return dump_filename;
}

void ObjectModeler::savePCDWithIndex(const std::string& filename_prefix, const CloudT& cloud, int index)
{
	fs::path dump_path = prepareDumpPath();
	fs::path dump_filename = dump_path / (filename_prefix + (boost::format("-%05d.pcd") % index).str());
#ifdef DISABLE_PCD_IO
				cout << "DISABLE_PCD_IO" << endl; exit(1);
#else
	pcl::io::savePCDFileBinary(dump_filename.string(), cloud); 
#endif
}

void ObjectModeler::enqueueShowImage(std::string name, cv::Mat image)
{
	boost::mutex::scoped_lock lock(mutex_show_image);
	queue_show_image.push_back(make_pair(name, image));
}

void ObjectModeler::showQueuedImages()
{
	boost::mutex::scoped_lock lock(mutex_show_image);
	while(!queue_show_image.empty()) {
		showInWindow(queue_show_image.front().first, queue_show_image.front().second);
		queue_show_image.pop_front();
	}
}

void ObjectModeler::processWaitKey()
{
	int k = cv::waitKey(params.process_frame_waitkey_delay);
	if (k >= 0) processKey(k);
}

void ObjectModeler::pauseAndShowImages()
{
	paused_by_user = true;
	while (paused_by_user) {
		showQueuedImages();
		processWaitKey();
	}
}

cv::Mat ObjectModeler::getPrettyDepthImage(cv::Mat raw_depth_image)
{
	cv::Mat nonzero_mask = raw_depth_image > 0;

	// get actual min / max
#if 0
	double min_z, max_z;
	int minInd, maxInd;
	cv::minMaxIdx(raw_depth_image, &min_z, &max_z, &minInd, &maxInd, nonzero_mask);
#endif
	// could also use camera params:
#if 0
	//double min_z = params.camera_z_min;
	//double max_z = params.camera_z_max;
#endif
	double min_z = params.display_min_z;
	double max_z = params.display_max_z;

	// either way
	double depth_range = max_z - min_z;

	cv::Mat scaled_depth = 1 - (raw_depth_image - min_z) / depth_range;
	cv::Mat result_float(raw_depth_image.size(), CV_32F, 0);
	scaled_depth.copyTo(result_float, nonzero_mask);
	return floatC1toCharC3(result_float);
}

bool ObjectModeler::processFrame(FrameT& frame)
{
	prepareFrame(frame);

	// always show input
	//cv::Mat depth_image_display = floatC1toCharC3(frame.image_depth * params.depth_scale_for_display);
	cv::Mat depth_image_display = getPrettyDepthImage(frame.image_depth);
	cv::Mat both_display = create1x2(frame.image_color, depth_image_display);
	showInWindow("input", both_display);
	if (params.save_input_images) {
		savePNGWithIndex("input_color", frame.image_color, input_frame_counter);
		savePNGWithIndex("input_depth", depth_image_display, input_frame_counter);
		savePNGWithIndex("input_both", both_display, input_frame_counter);
	}


	// if not active or not tracking but not empty, also overlay the last set of object points on the current frame for resume
	if ((!is_active || !is_tracking) && !isEmpty() && previous_object_pixels.data) {
		cv::Mat input_plus_previous_object = frame.image_color * 0.5 + previous_object_pixels * 0.5;
		// assume we want to flip for objects, but not for mapping
		if (params.mask_object) cv::flip(input_plus_previous_object, input_plus_previous_object, 1);
		showInWindow("input_plus_previous_object", input_plus_previous_object);
	}
	else {
		cv::destroyWindow("input_plus_previous_object");
	}

	// if command line params say to learn histogram and we aren't learning it, set it to learn (when activated)
	if (params.mask_hand_hist_learn && !learning_histogram_hand) {
		learning_histogram_hand = true;
		initialize_learn_histogram_next_frame = true;
	}
	if (params.mask_object_hist_learn && !learning_histogram_object) {
		learning_histogram_object = true;
		initialize_learn_histogram_next_frame = true;
	}


	///////////////////
	// main things to do with a frame:
	bool result = false;

	if (!is_active) {
		if (!isInLiveMode()) {
			cout << "Not active and not live mode..." << endl;
			return result;
		}
		static time_t last_warn = 0;
		time_t this_clock = clock();
		if ( (this_clock - last_warn) > 5.0 * CLOCKS_PER_SEC) {
			cout << "ObjectModeler not active" << endl;
			last_warn = this_clock;
		}
	}
	else if (params.folder_do_calibrate) {
		result = addFrameToCalibrate(frame);
	}
	else if (learning_histogram_hand || learning_histogram_object) {
		if (initialize_learn_histogram_next_frame) {
			initializeLearnHistogram();
			initialize_learn_histogram_next_frame = false;
		}
		result = learnHistogram(frame);
	}
	else if (params.save_input_only) {
		static pcl::StopWatch sw;
		static double last_save_time = -1;
		double current_time = sw.getTimeSeconds();
		double time_diff = current_time - last_save_time; // invalid if last_save_time < 0
		bool save_this_frame = true;
		if (params.save_input_only_max_fps > 0 && last_save_time > 0 && time_diff < 1.0 / params.save_input_only_max_fps) {
			cout << "Skipping frame due to params.save_input_only_max_fps: " << params.save_input_only_max_fps << endl;
			save_this_frame = false;
		}
		if (save_this_frame) {
			last_save_time = current_time;
			input_frame_counter++;
			savePCDWithIndex("input-cloud", *frame.cloud_ptr, input_frame_counter);
			float seconds = sw.getTimeSeconds();
			float fps = (float) input_frame_counter / sw.getTimeSeconds();
			cout << "Saved " << input_frame_counter << " frames in " << seconds << " seconds: " << fps << " FPS." << endl;
		}
	}
	else if (params.volume_debug_sphere) {
		setVolumeToDebugSphere();
		showVolumeEdges();
	}
	else {
		input_frame_counter++;

		// save input clouds, histograms, command line (so you can rerun)
		if (params.save_input) {
			pcl::ScopeTime st ("[TIMING] save_input_clouds");
			savePCDWithIndex("input-cloud", *frame.cloud_ptr, input_frame_counter);
			// also dump the hand and object histogram used...
			// simplest (and stupidest) is to always save....
			fs::path dump_path = prepareDumpPath();
			// dump to default filename
			fs::path dump_hand_hist = dump_path / params.mask_default_hand_hist_filename;
			fs::path dump_object_hist = dump_path / params.mask_default_object_hist_filename;
			saveHistogram(dump_hand_hist.string(), mask_object.histogram_hand);
			saveHistogram(dump_object_hist.string(), mask_object.histogram_object);
		}

		result = alignAndAddFrame(frame);
		is_tracking = result;

		if (is_tracking) {
			//  only set previous object pixels on success
			previous_object_pixels = cv::Mat::zeros(frame.image_color.size(), frame.image_color.type());
			frame.image_color.copyTo(previous_object_pixels, frame.object_mask);
		}

		if (params.set_inactive_on_failure && !result) {
			cout << "set_inactive_on_failure!" << endl;
			is_active = false;
		}

		if (params.save_tables_of_values) {
			writeTablesOfValues(prepareDumpPath());
		}

		if (params.save_viewer_screenshots) {
			save_screenshot_on_next_viewer_update = true;
			save_screenshot_counter = input_frame_counter;
		}
	}

	do {
		showQueuedImages();
		processWaitKey();
	} while (paused_by_user);

	return result;
}

void ObjectModeler::prepareFrame(FrameT& frame)
{
	pcl::StopWatch sw;

	if (params.max_depth_in_input_cloud > 0) {
		for (size_t i = 0; i < frame.cloud_ptr->size(); i++) {
			PointT& p = frame.cloud_ptr->at(i);
			if (p.z > params.max_depth_in_input_cloud) {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}

	if (params.mask_input_if_present && frame.object_mask.data) {
		for (int row = 0; row < frame.object_mask.rows; ++row) {
			for (int col = 0; col < frame.object_mask.cols; ++col) {
				if (!frame.object_mask.at<uchar>(row,col)) {
					PointT& p = frame.cloud_ptr->at(col,row);
					p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
				}
			}
		}
	}

	// hack the focal length if set
	if (params.correct_input_camera_params) {
		changeFocalLength(*frame.cloud_ptr, params.camera_focal_x, params.camera_focal_y, params.camera_center_x, params.camera_center_y);
	}

	// add all the images we'll need
	frame.addImagesToFrame();

	cout << "[TIMING] prepareFrame: " << sw.getTime() << endl;
}

bool ObjectModeler::addFrameToCalibrate(FrameT& frame)
{
	cv::Mat image_gray;
	cv::cvtColor(frame.image_color, image_gray, CV_BGR2GRAY);
	cv::Size patternsize(10,7);
	std::vector<cv::Point2f> corners;
	bool pattern_found = cv::findChessboardCorners(image_gray, patternsize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
	if (pattern_found) {
		cv::cornerSubPix(image_gray, corners, cv::Size(5,5), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	else {
		return false;
	}
	cv::drawChessboardCorners(image_gray, patternsize, cv::Mat(corners), pattern_found);
	showInWindow("corners", image_gray);

	// add corners to total and run calibration every time just to see
	// corners are bottom right to top left
	calibrate_object_points.push_back(std::vector<cv::Point3f>());
	float square_size = 1.0; // inches
	for (int row = 6; row >= 0; row--) {
		for (int col = 9; col >= 0; col--) {
			calibrate_object_points.back().push_back(cv::Point3d(col * square_size, row * square_size, 0));
		}
	}

	calibrate_image_points.push_back(corners);
	cv::Mat cameraMatrix;
	std::vector<double> distCoeffs;
	std::vector<std::vector<double> > rvecs;
	std::vector<std::vector<double> > tvecs;
	int flags = CV_CALIB_FIX_PRINCIPAL_POINT + CV_CALIB_ZERO_TANGENT_DIST + 
		CV_CALIB_FIX_K1 + CV_CALIB_FIX_K2 + CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 + CV_CALIB_FIX_K6;

	float reprojection_error = cv::calibrateCamera(calibrate_object_points, calibrate_image_points, image_gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs, flags);
	cout << "reprojection_error: " << reprojection_error << endl;

	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			cout << cameraMatrix.at<double>(row, col) << " ";
		}
		cout << endl;
	}

	// huh...about 540
	return true;
}

// no...single function, manipulate params outisde of this!
void ObjectModeler::initializeLearnHistogram()
{
	if (learning_histogram_hand) cout << "Learning hand histogram..." << endl;
	else if (learning_histogram_object) cout << "Learning object histogram..." << endl;
	else throw std::runtime_error ("HOW THE HELL?");

	while (true) {
		cout << "y: Reset histogram before learning" << endl;
		cout << "n: Add to existing histogram" << endl;
		int k_reset = cv::waitKey();
		if (k_reset == 'y') {
			learn_histogram.reset();
			break;
		}
		else if (k_reset == 'n') {
			if (learning_histogram_hand) learn_histogram.init(mask_object.histogram_hand);
			else if (learning_histogram_object) learn_histogram.init(mask_object.histogram_object);
			else throw std::runtime_error ("HOW THE HELL?");
			break;
		}
	}
}

bool ObjectModeler::learnHistogram(FrameT& frame)
{
	if (!(learning_histogram_hand || learning_histogram_object)) throw std::runtime_error ("!(learning_histogram_hand || learning_histogram_object)");
	if (learning_histogram_hand && learning_histogram_object) throw std::runtime_error ("learning_histogram_hand && learning_histogram_object");

	bool learn_result = learn_histogram.learn(frame);

	if (learn_result) {
		if (learning_histogram_hand) saveHistogram(params.mask_hand_hist_filename.string(), learn_histogram.getHistogram());
		else if (learning_histogram_object) saveHistogram(params.mask_object_hist_filename.string(), learn_histogram.getHistogram());
		else throw std::runtime_error ("HOW THE HELL DID YOU GET HERE!");
		// reload from disk for consistency...
		loadBothHistograms();
		learning_histogram_hand = false;
		learning_histogram_object = false;
		setActive(false);
	}

	return learn_result;
}

bool ObjectModeler::doObjectMasking(FrameT& frame)
{
	if (params.mask_object) {
		// real masking

		// set up search region
		cv::Rect seed_search_region = previous_object_rect;
		if (params.mask_always_reset_search_region) seed_search_region = cv::Rect();
		float contraction_factor = params.mask_search_contraction_factor;
		if (seed_search_region == cv::Rect()) {
			seed_search_region.x = 0;
			seed_search_region.y = 0;
			seed_search_region.width = frame.image_depth.cols;
			seed_search_region.height = frame.image_depth.rows;
			contraction_factor = params.mask_initial_search_contraction_factor;
		}
		seed_search_region = scaleRectCentered(seed_search_region, contraction_factor);

		bool foundObject = mask_object.addObjectCloudToFrame(frame, seed_search_region);
		previous_object_rect = frame.object_rect; // even if fail (resets seed search region) (could change this)

		// show these masks
		if (params.mask_debug_images) {
			showInWindow("object_mask", frame.object_mask);
		}

		// always show object pixels
		cv::Mat object_pixels (frame.object_mask.size(), CV_8UC3, cv::Scalar::all(128));
		frame.image_color.copyTo(object_pixels, frame.object_mask);
		showInWindow("object_pixels", object_pixels);

		// let debug images come up to date?
		if (params.mask_debug_images) processWaitKey();

		if (!foundObject) {
			cout << "\nFailed to find the object" << endl;
			return false;
		}
	}
	else {
		// fake masking (aka mapping mode)
		mask_object.fakeMasking(frame);
	}

	// debug 3d viewer
	tc_masked_points.setCloud(frame.object_cloud_ptr);

	if (params.save_objects_pcd) {
		// object cloud not full size
		CloudT::Ptr object_cloud_full_size = maskCloud(*frame.cloud_ptr, frame.object_mask);
		savePCDWithIndex("object-cloud", *object_cloud_full_size, input_frame_counter);
	}

	if (params.save_objects_png) {
		savePNGWithIndex("object-image", frame.image_color, input_frame_counter);
		// get uint16 depth image masked to object in the stupidest way possible
		CloudT::Ptr object_cloud_full_size = maskCloud(*frame.cloud_ptr, frame.object_mask);
		cv::Mat object_depth_image_float = cloudToDepthImage(*object_cloud_full_size);
		for (int row = 0; row < object_depth_image_float.rows; ++row) {
			for (int col = 0; col < object_depth_image_float.cols; ++col) {
				float& float_value = object_depth_image_float.at<float>(row,col); 
				if (pcl_isnan(float_value)) {
					float_value = 0;
				}
				if (float_value < 0) {
					throw std::runtime_error ("float_value < 0");
				}
			}
		}
		cv::Mat object_depth_image_uint16;
		object_depth_image_float.convertTo(object_depth_image_uint16, CV_16UC1, 1000);
		savePNGWithIndex("object-depth", object_depth_image_uint16, input_frame_counter);
	}

	return true;
}

void ObjectModeler::copyNormalsFromImageBuffers(FrameT& frame)
{
	// assume we've copied already
	if (frame.object_normal_cloud_ptr) return;

	frame.object_normal_cloud_ptr.reset(new pcl::PointCloud<pcl::Normal>);
	frame.object_normal_cloud_ptr->width = frame.object_rect.width;
	frame.object_normal_cloud_ptr->height = frame.object_rect.height;
	frame.object_normal_cloud_ptr->is_dense = false;
	frame.object_normal_cloud_ptr->resize(frame.object_normal_cloud_ptr->width * frame.object_normal_cloud_ptr->height);

	std::vector<float> normals_vector(4 * frame.object_normal_cloud_ptr->size());
	frame.image_buffer_normals_ptr->readToFloatVector(normals_vector);
	for (int i = 0; i < frame.object_normal_cloud_ptr->size(); ++i) {
		pcl::Normal& n = frame.object_normal_cloud_ptr->at(i);
		n.normal_x = normals_vector[4 * i];
		n.normal_y = normals_vector[4 * i + 1];
		n.normal_z = normals_vector[4 * i + 2];
	}

	frame.object_cloud_normal_mask = getMaskOfValidNormals(*frame.object_normal_cloud_ptr);
}

void ObjectModeler::addNormalsToFrame(FrameT& frame)
{
	CloudT::Ptr cloud = frame.object_cloud_ptr;

	// first copy points to the buffer
	std::vector<float> points_vector;
	points_vector.resize(4 * cloud->size());
	for (int i = 0; i < cloud->size(); ++i) {
		const PointT& p = cloud->at(i);
		points_vector[4 * i] = p.x;
		points_vector[4 * i + 1] = p.y;
		points_vector[4 * i + 2] = p.z;
		points_vector[4 * i + 3] = 1;
	}

	// then compute on the GPU (leave them there)
	frame.image_buffer_points_ptr.reset(new ImageBuffer(*cl_ptr));
	frame.image_buffer_normals_ptr.reset(new ImageBuffer(*cl_ptr));
	frame.image_buffer_points_ptr->writeFromFloatVector(points_vector);
	opencl_normals_ptr->computeNormalsWithBuffers(cloud->width, cloud->height, *frame.image_buffer_points_ptr, params.max_depth_sigmas, params.normals_smooth_iterations, *frame.image_buffer_normals_ptr);

	if (params.normals_opencl_debug) {
		// bring back to CPU
		copyNormalsFromImageBuffers(frame);

		pcl::PointCloud<pcl::Normal>::Ptr compare_cloud = computeNormals(cloud);

		cv::Mat image_opencl = normalCloudToRGBImage(*frame.object_normal_cloud_ptr);
		cv::Mat image_compare = normalCloudToRGBImage(*compare_cloud);
		cv::Mat image_diff = (image_opencl - image_compare) + (image_compare - image_opencl);
		std::vector<cv::Mat> v_images;
		v_images.push_back(image_opencl);
		v_images.push_back(image_compare);
		v_images.push_back(image_diff);
		cv::Mat all = createMxN(1,3,v_images);
		showInWindow("normals_opencl_debug", all);
	}
}

bool ObjectModeler::alignPreparedFrame(FrameT& frame)
{
	bool result = false;
	const bool is_first_frame = isEmpty();

	// "hack" to load existing poses
	if (!input_object_poses.empty()) {
		int pose_index = input_frame_counter - 1; // assumes input_frame_count >= 1 by this point
		if (!input_object_poses.at(pose_index).first) {
			cout << "No pose in file for pose_index: " << pose_index << endl;
			return false;
		}
		object_pose = input_object_poses[pose_index].second;
		// set up other stuff?
		return true;
	}	

	if (is_first_frame) {
		result = true;

		// put the object pose at the centroid of object points
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*frame.object_cloud_ptr, centroid);
		if (params.initial_centroid_fixed) {
			centroid.x() = centroid.y() = centroid.z() = 0;
		}
		else if (params.initial_centroid_image_center) {
			centroid.x() = centroid.y() = 0;
		}
		object_pose.setIdentity();
		// origin is now 0,0,0
		Eigen::Vector3f centering_translation(
			-params.volume_cell_count_x * params.volume_cell_size / 2,
			-params.volume_cell_count_y * params.volume_cell_size / 2,
			-params.volume_cell_count_z * params.volume_cell_size / 2);
		object_pose = Eigen::Translation3f(centering_translation);
		object_pose.pretranslate(centroid.head<3>() + Eigen::Vector3f(params.initial_centroid_offset_x, params.initial_centroid_offset_y, params.initial_centroid_offset_z));
		previous_relative_transform = Eigen::Affine3f::Identity(); // don't want this first jump to count

		// in case of frame to model
		if (params.use_features) {
			std::vector<cv::DMatch> empty_inliers;
			updateModelKeypoints(object_pose, frame, empty_inliers);
		}
		else {
			// a bit of a hack to keep later code from crashing
			model_kp.keypoint_cloud.reset(new KeypointCloudT);
		}
	}
	else {
		Eigen::Affine3f previous_object_pose = object_pose;

		if (params.initialize_with_previous_relative_transform) {
			object_pose = previous_relative_transform * object_pose;
		}

		// need these for later:
		std::vector<cv::DMatch> inlier_matches;
		bool success_features = false;
		bool success_icp = false;
		if (params.use_features) {
			Eigen::Affine3f feature_result_pose;
			success_features = alignWithFeaturesToModel(object_pose, frame, model_kp, feature_result_pose, inlier_matches);
			globalStopWatchMark("after alignWithFeaturesToModel");
			if (success_features) {
				object_pose = feature_result_pose;
			}
			else {
				cout << "Failed to align using features...object pose not modified" << endl;
			}
		}
		std::vector<cv::DMatch> trusted_inlier_matches = inlier_matches;
		if (!success_features) trusted_inlier_matches.clear();

		// Combined Optimization (ICP + color)
		if (params.use_combined_optimization) {
			Eigen::Affine3f icp_result_pose;

			RenderBuffers render_buffers(*cl_ptr);
			cv::Rect render_rect;
			globalStopWatchMark("before renderForAlign");
			bool render_ok = renderForAlign(frame, false, object_pose, render_buffers, render_rect);
			globalStopWatchMark("after renderForAlign");
			if (render_ok) {
				Eigen::Affine3f initial_relative_pose = Eigen::Affine3f::Identity();
				success_icp = alignWithCombinedOptimizationNew(frame, render_buffers, render_rect, params.combined_debug_images, object_pose, initial_relative_pose, icp_result_pose);
			}
			globalStopWatchMark("after alignWithCombinedOptimizationNew");

			if (success_icp) {
				object_pose = icp_result_pose;
			}
			else {
				cout << "Failed to alignWithCombinedOptimization...object pose not modified" << endl;
			}
		}

		result = success_features || success_icp;

		if (result) {
			previous_relative_transform = object_pose * previous_object_pose.inverse();

			// we update keypoints after ICP refinement
			if (params.use_features) {
				updateModelKeypoints(object_pose, frame, trusted_inlier_matches);
			}
		}
		else {
			previous_relative_transform.setIdentity();
		}
	}

	return result;
}

void ObjectModeler::addFrameToModel(FrameT& frame)
{
	if (params.use_patch_grid) {
		updateGrid(frame);
	}
	else if (params.use_patch_volumes) {
		// needs normals on CPU
		copyNormalsFromImageBuffers(frame);

		updatePatchVolumes(frame);
	}
	else {
		// single volume
		cv::Mat object_depth_image(frame.image_depth.size(), frame.image_depth.type(), cv::Scalar(0));
		frame.image_depth.copyTo(object_depth_image, frame.object_mask);
		cv::Mat image_color_uchar4;
		cv::cvtColor(frame.image_color, image_color_uchar4, CV_BGR2BGRA);

		ImageBuffer buffer_depth_image(*cl_ptr);
		ImageBuffer buffer_color_image(*cl_ptr);
		ImageBuffer buffer_segments(*cl_ptr);
		buffer_depth_image.writeFromBytePointer(object_depth_image.data, object_depth_image.total() * sizeof(float));
		buffer_color_image.writeFromBytePointer(image_color_uchar4.data, image_color_uchar4.total() * 4 * sizeof(uint8_t));
		// can leave buffer_segments empty if which_segment = 0;
		opencl_tsdf_ptr->addFrame(object_pose, buffer_depth_image, buffer_color_image, buffer_segments, 0);
	}

	output_frame_counter++;
}

void ObjectModeler::addKeyframe(FrameT const& frame, std::vector<int> const& updated_this_frame)
{
	Eigen::Affine3f frame_pose = camera_list.back()->pose;
	bool add_keyframe = false;
	bool is_first_keyframe = keyframes.empty();
	if (is_first_keyframe) {
		add_keyframe = true;
	}
	else {
		Eigen::Affine3f last_keyframe_pose = vertex_to_camera_map[keyframes.back().vertex_id]->pose;
		Eigen::Affine3f relative_pose = frame_pose * last_keyframe_pose.inverse();
		float translation = relative_pose.translation().norm();
		Eigen::AngleAxisf aa(relative_pose.rotation());
		float angle_degrees = 180.f / M_PI * aa.angle();

		if (translation > params.pv_keyframe_max_distance_create || angle_degrees > params.pv_keyframe_max_angle_create) {
			add_keyframe = true;
		}
	}
	if (add_keyframe) {
		KeyframeStruct new_keyframe;
		new_keyframe.frame_ptr.reset(new FrameT(frame));
		/////
		// save memory by only keeping color image + features
		new_keyframe.frame_ptr->cloud_ptr.reset();
		//new_keyframe.frame_ptr->image_color = cv::Mat();
		new_keyframe.frame_ptr->image_color_hsv = cv::Mat();
		new_keyframe.frame_ptr->image_depth = cv::Mat();
		new_keyframe.frame_ptr->depth_mask = cv::Mat();
		new_keyframe.frame_ptr->depth_mask_without_hand = cv::Mat();
		new_keyframe.frame_ptr->object_mask = cv::Mat();
		new_keyframe.frame_ptr->object_cloud_normal_mask = cv::Mat();
		// note we keep the object_kp set
		new_keyframe.frame_ptr->object_cloud_ptr.reset();
		new_keyframe.frame_ptr->object_normal_cloud_ptr.reset();
		new_keyframe.frame_ptr->object_kp_projection_cloud_ptr.reset(); // note we don't need projections for keypoints
		////
		new_keyframe.vertex_id = camera_list.back()->vertex_id;
		keyframes.push_back(new_keyframe);
		for (int i = 0; i < updated_this_frame.size(); ++i) {
			pv_to_keyframes_map[updated_this_frame[i]].insert(keyframes.size() - 1);
		}

		if (params.pv_keyframe_debug) {
			showInWindow("keyframe", keyframes.back().frame_ptr->image_color);
			processWaitKey();
		}
	}
}

bool ObjectModeler::alignAndAddFrame(FrameT& frame)
{
	// new: lock volumes for whole add and align
	boost::mutex::scoped_lock lock_volume(mutex_volumes);

	pcl::StopWatch sw;
	g_sw.reset();

	if (!doObjectMasking(frame)) return false;
	globalStopWatchMark("masking");

	if (params.use_features || params.pv_loop_features) addObjectFeaturesToFrame(frame);
	globalStopWatchMark("compute features");

	addNormalsToFrame(frame);
	//rs_add_normals_to_frame_outside.push(g_sw.getTime());
	//cout << "rs_add_normals_to_frame_outside: " << rs_add_normals_to_frame_outside.summary() << endl;
	globalStopWatchMark("compute normals");

	globalStopWatchMark("before align prepared frame");
	bool ok_to_add_frame = alignPreparedFrame(frame);
	globalStopWatchMark("after align prepared frame");

	//////////
	// Inject pose saving here
	if (params.save_poses) {
		if (ok_to_add_frame) {
			ofs_object_pose << EigenUtilities::transformToString(object_pose) << endl;
			ofs_camera_pose << EigenUtilities::transformToString(object_pose.inverse()) << endl;
		}
		else {
			// blank line means fail
			ofs_object_pose << endl;
			ofs_camera_pose << endl;
			any_failures_save_exception = true; // if we are saving loop closure poses, they will be wrong (currently, until I fix it)
		}
	}

	if (ok_to_add_frame) {
		addFrameToModel(frame);
	}
	globalStopWatchMark("add frame to model");

	if (!params.use_patch_volumes) {
		showVolumeEdges();
		globalStopWatchMark("viewer debug (showVolumeEdges)");
	}
	
	if (params.render_after) {
		renderAfterAlign(frame, object_pose);
		globalStopWatchMark("renderAfterAlign");
	}

	rs_addAndAlign.push(sw.getTime());
	cout << "Overall Time per frame (ms): " << rs_addAndAlign.summary() << endl;
	cout << (boost::format("Added %d/%d frames") % output_frame_counter % input_frame_counter).str() << endl;
	cout << "Time / pvs actually rendered: " << rs_addAndAlign.lastValue() / rs_pvs_rendered.lastValue() << endl;

	// return value should say whether we are still tracking well
	return ok_to_add_frame;
}

CloudT::Ptr ObjectModeler::getLineSetEdges(const std::vector<Eigen::Vector3f>& corners) const
{
	cv::Vec3b line_color(255,255,255); // assume white
	if (params.white_background) {
		line_color = cv::Vec3b(0,0,0); // except black on white
	}
	return getLineSetEdges(corners, line_color);
}

CloudT::Ptr ObjectModeler::getLineSetEdges(const std::vector<Eigen::Vector3f>& corners, const cv::Vec3b& color) const
{
	// lines for the volume
	CloudT::Ptr corner_line_cloud(new CloudT);
	// uses knowledge that corners are in "binary" order (000, 001, 010, 011, etc.)
	corner_line_cloud->points.resize(24); // 12 lines
	// "0xx" face:
	corner_line_cloud->points[0].getVector3fMap() = corners[0];
	corner_line_cloud->points[1].getVector3fMap() = corners[1];
	corner_line_cloud->points[2].getVector3fMap() = corners[1];
	corner_line_cloud->points[3].getVector3fMap() = corners[3];
	corner_line_cloud->points[4].getVector3fMap() = corners[3];
	corner_line_cloud->points[5].getVector3fMap() = corners[2];
	corner_line_cloud->points[6].getVector3fMap() = corners[2];
	corner_line_cloud->points[7].getVector3fMap() = corners[0];
	// "1xx" face:
	int corner_cloud_offset = 8;
	int corner_vector_offset = 4;
	corner_line_cloud->points[0+corner_cloud_offset].getVector3fMap() = corners[0+corner_vector_offset];
	corner_line_cloud->points[1+corner_cloud_offset].getVector3fMap() = corners[1+corner_vector_offset];
	corner_line_cloud->points[2+corner_cloud_offset].getVector3fMap() = corners[1+corner_vector_offset];
	corner_line_cloud->points[3+corner_cloud_offset].getVector3fMap() = corners[3+corner_vector_offset];
	corner_line_cloud->points[4+corner_cloud_offset].getVector3fMap() = corners[3+corner_vector_offset];
	corner_line_cloud->points[5+corner_cloud_offset].getVector3fMap() = corners[2+corner_vector_offset];
	corner_line_cloud->points[6+corner_cloud_offset].getVector3fMap() = corners[2+corner_vector_offset];
	corner_line_cloud->points[7+corner_cloud_offset].getVector3fMap() = corners[0+corner_vector_offset];
	// between "0xx" and "1xx"
	corner_cloud_offset += 8;
	corner_line_cloud->points[0+corner_cloud_offset].getVector3fMap() = corners[0];
	corner_line_cloud->points[1+corner_cloud_offset].getVector3fMap() = corners[4];
	corner_line_cloud->points[2+corner_cloud_offset].getVector3fMap() = corners[1];
	corner_line_cloud->points[3+corner_cloud_offset].getVector3fMap() = corners[5];
	corner_line_cloud->points[4+corner_cloud_offset].getVector3fMap() = corners[2];
	corner_line_cloud->points[5+corner_cloud_offset].getVector3fMap() = corners[6];
	corner_line_cloud->points[6+corner_cloud_offset].getVector3fMap() = corners[3];
	corner_line_cloud->points[7+corner_cloud_offset].getVector3fMap() = corners[7];

	// set color for all points
	for (CloudT::iterator i = corner_line_cloud->begin(); i != corner_line_cloud->end(); ++i) {
		i->r = color[2];
		i->g = color[1];
		i->b = color[0];
	}

	return corner_line_cloud;
}

boost::shared_ptr<std::vector<pcl::Vertices> > ObjectModeler::getMeshVerticesForCorners(int offset)
{
	boost::shared_ptr<std::vector<pcl::Vertices> > result(new std::vector<pcl::Vertices>(12));
	uint32_t fc = 0;
	// left
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(1+offset);
	result->at(fc++).vertices.push_back(3+offset);
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(3+offset);
	result->at(fc++).vertices.push_back(2+offset);

	// right
	result->at(fc).vertices.push_back(4+offset);
	result->at(fc).vertices.push_back(7+offset);
	result->at(fc++).vertices.push_back(5+offset);
	result->at(fc).vertices.push_back(4+offset);
	result->at(fc).vertices.push_back(6+offset);
	result->at(fc++).vertices.push_back(7+offset);

	// front
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(2+offset);
	result->at(fc++).vertices.push_back(6+offset);
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(6+offset);
	result->at(fc++).vertices.push_back(4+offset);

	// back 
	result->at(fc).vertices.push_back(1+offset);
	result->at(fc).vertices.push_back(5+offset);
	result->at(fc++).vertices.push_back(7+offset);
	result->at(fc).vertices.push_back(1+offset);
	result->at(fc).vertices.push_back(7+offset);
	result->at(fc++).vertices.push_back(3+offset);

	// top
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(5+offset);
	result->at(fc++).vertices.push_back(1+offset);
	result->at(fc).vertices.push_back(0+offset);
	result->at(fc).vertices.push_back(4+offset);
	result->at(fc++).vertices.push_back(5+offset);

	// bottom
	result->at(fc).vertices.push_back(2+offset);
	result->at(fc).vertices.push_back(3+offset);
	result->at(fc++).vertices.push_back(7+offset);
	result->at(fc).vertices.push_back(2+offset);
	result->at(fc).vertices.push_back(7+offset);
	result->at(fc++).vertices.push_back(6+offset);

	return result;
}

void ObjectModeler::combinedDebugImagesNew(FrameT const& frame, Eigen::Affine3f const& transform, int image_channel_count,
		std::vector<float> const& error_vector, std::vector<float> const& weight_vector,
		cv::Rect const& render_rect,
		std::vector<cv::Mat> const& frame_images_full_size, cv::Rect const& object_rect,
		RenderBuffers const& render_buffers, Eigen::Vector2f const& render_proj_f, Eigen::Vector2f const& render_proj_c)
{
	const int rows = render_rect.height;
	const int cols = render_rect.width;
	int error_points = rows * cols;
	int error_channel_count = image_channel_count + 1;

	// init "top row" error images to 0.5
	std::vector<cv::Mat> image_error_vec;
	for (int i = 0; i < error_channel_count; i++) {
		image_error_vec.push_back(cv::Mat(render_rect.height, render_rect.width, CV_32FC1, cv::Scalar::all(0.5)));
	}

	// fiddle with the "top row"
	std::vector<float> sse_vec(error_channel_count, 0);
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			// new coallesced indexing (TODO: JUST USE OPENCV)
			// this lets me grab the actual error, though!
			// but what you should do is start with actual error, then later add 0.5
			float sse = 0;
			for (int i = 0; i < error_channel_count; i++) {
				int error_vector_index_image = i * error_points + row * cols + col;
				float error_value = error_vector[error_vector_index_image];
				sse_vec[i] += error_value * error_value;
				image_error_vec[i].at<float>(row, col) += error_value;
			}
		}
	}

	// bottom row is weights errors now....
	std::vector<cv::Mat> image_weights_vec;
	for (int i = 0; i < error_channel_count; i++) {
		image_weights_vec.push_back(cv::Mat(render_rect.height, render_rect.width, CV_32FC1, cv::Scalar::all(0)));
	}
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			// new coallesced indexing (TODO: JUST USE OPENCV)
			for (int i = 0; i < error_channel_count; i++) {
				int index = i * error_points + row * cols + col;
				float weight = weight_vector[index];
				image_weights_vec[i].at<float>(row, col) = weight;
			}
		}
	}



	// we can now print the total error per image
	// TODO: COMPUTE VARIANCE/STD OF ERROR!!!
	for (int i = 0; i < image_error_vec.size(); ++i) {
		//float total_squared_error = cv::sum(image_error_vec[i].mul(image_error_vec[i]))[0];
		cout << "Sum Squared Error " << i << " : " << sse_vec[i] << endl;
		// also "variance"
		float variance = sse_vec[i] / render_rect.area();
		cout << "Variance " << i << " : " << variance << endl;
		cout << "STD " << i << " : " << sqrt(variance) << endl;
	}

	// also get a render mask to show missing values
	cv::Mat render_mask_int(render_rect.height, render_rect.width, CV_32S);
	render_buffers.readRenderMask((int*)render_mask_int.data);
	cv::Mat render_mask = render_mask_int > 0;

#if 0
	// get reference images for bottom row (note: image_channel_count not error_channel_count)
	std::vector<cv::Mat> frame_image_channels;
	for (int i = 0; i < frame_images_full_size.size(); ++i) {
		frame_image_channels.push_back(frame_images_full_size[i](object_rect));
	}

	// The old row 2
	for (int i = 0; i < error_channel_count; i++) {
		// scale frame images to match render resolution here
		cv::Mat source_image = frame_image_channels[i];
		cv::Mat image_to_show = source_image;
		cv::Size desired_size = v_images[0].size();
		if (source_image.size() != desired_size) {
			cv::resize(source_image, image_to_show, desired_size, 0, 0, cv::INTER_NEAREST);
		}
		v_images.push_back(floatC1toCharC3(image_to_show));
	}

#endif

	// split this into cloud creation once and then an overlaty function (for loop closure di as well)
#if 0
	// make the cloud overlay image
	// need a cloud with color (will be slow)
	// this is silly...should take cloud as argument once (this is called many times with same cloud)
	cout << "YOU ARE RUNNING SILLY SLOW CODE" << endl;
	std::vector<float> render_points(render_rect.area() * 4);
	render_buffers.readRenderPoints(render_points.data());
	std::vector<uchar> render_colors(render_rect.area() * 4);
	render_buffers.readRenderColorImage(render_colors.data());
	CloudT::Ptr render_cloud(new CloudT);
	render_cloud->resize(render_rect.area());
	render_cloud->height = render_rect.height;
	render_cloud->width = render_rect.width;
	for (int i = 0; i < render_cloud->size(); ++i) {
		PointT & p = render_cloud->at(i);
		p.x = render_points[4*i];
		p.y = render_points[4*i+1];
		p.z = render_points[4*i+2];
		p.b = render_colors[4*i];
		p.g = render_colors[4*i+1];
		p.r = render_colors[4*i+2];
	}

	// project the point cloud by transform
	CloudT::Ptr transformed_render_cloud(new CloudT);
	pcl::transformPointCloud(*render_cloud, *transformed_render_cloud, transform);
	CloudT::Ptr projected_cloud = projectRenderCloud(*transformed_render_cloud, render_proj_f, render_proj_c, Eigen::Vector2f(render_rect.x, render_rect.y));
	cv::Mat projected_image = cloudToImage(*projected_cloud);

	// This is the maybe useful result:
	cv::Mat projected_on_frame = frame.image_color(render_rect) * 0.5 + projected_image * 0.5;
#endif


	// fill the v_images
	// row 1
	std::vector<cv::Mat> v_images;
	for (int i = 0; i < error_channel_count; i++) {
		cv::Mat error_image_bgr = floatC1toCharC3(image_error_vec[i]);
		cv::Mat error_image_bgr_masked (rows, cols, CV_8UC3, cv::Scalar(0,0,255)); // red outliers?
		error_image_bgr.copyTo(error_image_bgr_masked, render_mask);

		// also outliers (those errors which are exactly 0.5)
		cv::Mat outlier_mask = render_mask & image_error_vec[i] == 0.5;
		error_image_bgr_masked.setTo(cv::Scalar(255,0,0), outlier_mask);

		v_images.push_back(error_image_bgr_masked);
	}
	// row 2
	for (int i = 0; i < error_channel_count; i++) {
		cv::Mat weight_image_bgr = floatC1toCharC3(image_weights_vec[i]);
		v_images.push_back(weight_image_bgr);
	}
	cv::Mat combined_error_images = createMxN(2, error_channel_count, v_images);
	float scale = params.combined_debug_images_scale;
	cv::Mat combined_error_images_scaled;
	cv::resize(combined_error_images, combined_error_images_scaled, cv::Size(), scale, scale, cv::INTER_NEAREST); 
	showInWindow("Combined Error Images (new)", combined_error_images_scaled);

	if (params.save_cdi_images) {
		static int debug_image_counter = 0;

		// projected + color
//		cv::Mat overlay_error = create1x2(projected_on_frame, floatC1toCharC3(image_error_vec[1]));
//		savePNGWithIndex("overlay_error", overlay_error, debug_image_counter);

		// also 4-way
		savePNGWithIndex("overlay_error_4_way", combined_error_images, debug_image_counter);

		debug_image_counter++;
	}

	if (params.combined_pause_every_eval) pauseAndShowImages();
	else cv::waitKey(1);
}

size_t ObjectModeler::getImageChannelCount() const
{
	if (params.combined_image_error == params.IMAGE_ERROR_YCBCR) return 3;
	else if (params.combined_image_error == params.IMAGE_ERROR_CBCR) return 2;
	else if (params.combined_image_error == params.IMAGE_ERROR_Y) return 1;
	else if (params.combined_image_error == params.IMAGE_ERROR_LAB) return 3;
	else if (params.combined_image_error == params.IMAGE_ERROR_NONE) return 0;

	throw std::runtime_error ("NOT IMPLEMENTED");
}

std::vector<ImageBuffer> ObjectModeler::getImageChannelsList(ImageBuffer const& color_bgra_uchar, size_t width, size_t height)
{
	std::vector<ImageBuffer> result;

	if (params.combined_image_error == params.IMAGE_ERROR_YCBCR) {
		result.push_back(opencl_images_ptr->extractYFloat(color_bgra_uchar, width, height));
		result.push_back(opencl_images_ptr->extractCrFloat(color_bgra_uchar, width, height));
		result.push_back(opencl_images_ptr->extractCbFloat(color_bgra_uchar, width, height));
	}
	else if (params.combined_image_error == params.IMAGE_ERROR_CBCR) {
		result.push_back(opencl_images_ptr->extractCrFloat(color_bgra_uchar, width, height));
		result.push_back(opencl_images_ptr->extractCbFloat(color_bgra_uchar, width, height));
	}
	else if (params.combined_image_error == params.IMAGE_ERROR_Y) {
		result.push_back(opencl_images_ptr->extractYFloat(color_bgra_uchar, width, height));
	}
	else if (params.combined_image_error == params.IMAGE_ERROR_LAB) {
		// do on cpu like a loser
		// note this puts on GPU, then brings off, then does this stuff...really bad
		cv::Mat mat_color_bgra_uchar(height, width, CV_8UC4);
		color_bgra_uchar.readToBytePointer(mat_color_bgra_uchar.data, width * height * 4);
		cv::Mat mat_color_bgr_uchar;
		cv::cvtColor(mat_color_bgra_uchar, mat_color_bgr_uchar, CV_BGRA2BGR);
		cv::Mat mat_lab_uchar;
		cv::cvtColor(mat_color_bgr_uchar, mat_lab_uchar, CV_BGR2Lab);
		cv::Mat mat_lab_float;
		mat_lab_uchar.convertTo(mat_lab_float, CV_32F, 1./255.);
		std::vector<cv::Mat> lab_split;
		cv::split(mat_lab_float, lab_split);
		for (int i = 0; i < 3; ++i) {
			result.push_back(ImageBuffer(*cl_ptr));
			result.back().reallocateIfNeeded(width * height * sizeof(float));
			result.back().writeFromBytePointer(lab_split[i].data, width * height * sizeof(float));
		}
	}
	else if (params.combined_image_error == params.IMAGE_ERROR_NONE) {
		// nothing!	
	}
	else {
		throw std::runtime_error ("NOT IMPLEMENTED");
	}

	return result;
}

ImageBuffer ObjectModeler::packImageChannelsList(std::vector<ImageBuffer> const& image_buffer_list, size_t width, size_t height)
{
	ImageBuffer result(*cl_ptr);
	size_t image_size_single_float = width * height * sizeof(float);
	result.reallocateIfNeeded(image_size_single_float * image_buffer_list.size() );
	for (int i = 0; i < image_buffer_list.size(); ++i) {
		cl_ptr->queue.enqueueCopyBuffer(image_buffer_list[i].getBuffer(), result.getBuffer(), 0, image_size_single_float * i, image_size_single_float);
	}
	return result;
}

void ObjectModeler::showAndSaveRenderDebugImages(FrameT const& frame,
	std::string name, 
	bool do_save,
	RenderBuffers & render_buffers,
	cv::Rect& render_rect)
{
	// make renger_image_bgr
	cv::Mat render_image_bgra(render_rect.height, render_rect.width, CV_8UC4);
	render_buffers.readRenderColorImage(render_image_bgra.data);
	cv::Mat render_image_bgr;
	cv::cvtColor(render_image_bgra, render_image_bgr, CV_BGRA2BGR);

	// and normals
	std::vector<float> render_normals(render_rect.height * render_rect.width * 4);
	render_buffers.readRenderNormals(render_normals.data());
	// make a cloud to use existing functions
	pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
	normal_cloud->width = render_rect.width;
	normal_cloud->height = render_rect.height;
	normal_cloud->resize(normal_cloud->width * normal_cloud->height);
	for (int i = 0; i < normal_cloud->size(); ++i) {
		pcl::Normal & n = normal_cloud->at(i);
		n.normal_x = render_normals[4*i];
		n.normal_y = render_normals[4*i+1];
		n.normal_z = render_normals[4*i+2];
	}
	cv::Mat normals_image_bgr = normalCloudToImage(*normal_cloud, light_direction);

	// show both
	cv::Mat both = create1x2(render_image_bgr, normals_image_bgr);

	showInWindow(name, both);

	if (do_save) {
		//savePNGWithIndex(name, render_image_bgr, input_frame_counter);

#if 0
		// assume we want 640x480 always
		cv::Mat render_image_640 (480, 640, CV_8UC3, cv::Scalar::all(0));
		render_image_bgr.copyTo(render_image_640(render_rect));
		savePNGWithIndex(name+"_640", render_image_640, input_frame_counter);
#endif

		// also assume we want both
		savePNGWithIndex(name+"_both", both, input_frame_counter);

		// todo: might want both at 640 if doing objects
		///////

		cv::Mat input_color = frame.image_color;
		cv::Mat input_depth = getPrettyDepthImage(frame.image_depth);

		cv::Mat frame_normals_lit = normalCloudToImage(*frame.object_normal_cloud_ptr, light_direction);

		std::vector<cv::Mat> image_v;
		image_v.push_back(input_color);
		//image_v.push_back(input_depth);
		image_v.push_back(frame_normals_lit);
		image_v.push_back(render_image_bgr);
		image_v.push_back(normals_image_bgr);
		cv::Mat four_way = createMxN(2, 2, image_v);
		savePNGWithIndex(name + "_four_way", four_way, input_frame_counter);
	}
}

void ObjectModeler::renderAfterAlign(FrameT const& frame,
	Eigen::Affine3f const& pose)
{
	// could be result args:
	RenderBuffers render_buffers(*cl_ptr);
	cv::Rect render_rect = frame.object_rect;

	// some duplicate...hacky:
	const Eigen::Vector2f render_proj_f(params.camera_focal_x, params.camera_focal_y);
	const Eigen::Vector2f render_proj_c(params.camera_center_x, params.camera_center_y);
	float render_min_depth = params.camera_z_min;
	float render_max_depth = params.camera_z_max;

	if (params.use_patch_volumes) {
		int pv_min_age = -1;
		int pv_max_age = params.pv_max_age_before_considered_loop;
		float pv_max_normal_angle = -1;
		bool deallocate_after = false;
		bool update_frame_in_frustum = false;

		renderPatchVolumes(pose, render_proj_f, render_proj_c, render_min_depth, render_max_depth, render_rect, pv_min_age, pv_max_age, pv_max_normal_angle, deallocate_after, update_frame_in_frustum, render_buffers);
	}
	else {
		renderVolumeWithOpenCL(pose, render_proj_f, render_proj_c, render_min_depth, render_max_depth, render_rect, render_buffers);
	}

	std::string name = "render_after_align";
	showAndSaveRenderDebugImages(frame, name, params.save_render_after, render_buffers, render_rect);
}

bool ObjectModeler::renderForAlign(FrameT const& frame,
	bool is_loop_closure,
	Eigen::Affine3f const& initial_pose,
	RenderBuffers & render_buffers,
	cv::Rect & render_rect)
{
	// assume these are const throughout
	const Eigen::Vector2f render_proj_f(params.camera_focal_x, params.camera_focal_y);
	const Eigen::Vector2f render_proj_c(params.camera_center_x, params.camera_center_y);
	float render_min_depth = params.camera_z_min;
	float render_max_depth = params.camera_z_max;

	render_rect = frame.object_rect;

	if (params.use_patch_volumes) {
		int pv_min_age = is_loop_closure ? (params.pv_max_age_before_considered_loop + 1) : -1;
		int pv_max_age = is_loop_closure ? -1 : params.pv_max_age_before_considered_loop;
		float pv_max_normal_angle = is_loop_closure ? params.pv_loop_max_normal_angle : -1;
		bool deallocate_after = is_loop_closure ? true : false;
		bool update_frame_in_frustum = is_loop_closure ? false : true;

		renderPatchVolumes(initial_pose, render_proj_f, render_proj_c, render_min_depth, render_max_depth, render_rect, pv_min_age, pv_max_age, pv_max_normal_angle, deallocate_after, update_frame_in_frustum, render_buffers);
	}
	else {
		renderVolumeWithOpenCL(initial_pose, render_proj_f, render_proj_c, render_min_depth, render_max_depth, render_rect, render_buffers);
	}


	if (params.combined_show_render) {
		std::string name = is_loop_closure ? "render_for_align_loop" : "render_for_align_sequential";
		showAndSaveRenderDebugImages(frame, name, params.save_render_for_alignment, render_buffers, render_rect);
	}

	// need to check point counts for result
	cv::Mat render_mask(render_rect.height, render_rect.width, CV_32S);
	render_buffers.readRenderMask((int*)render_mask.data);
	int non_zero = cv::countNonZero(render_mask);

	if (non_zero < params.combined_min_rendered_point_count) {
		cout << "params.combined_min_rendered_point_count " << params.combined_min_rendered_point_count << " not met by " << non_zero << endl;
		return false;
	}

	if (is_loop_closure) {
		float render_coverage = (float)non_zero / (float)render_rect.area();
		if (render_coverage < params.pv_loop_min_frame_coverage) {
			cout << "params.pv_loop_min_frame_coverage " << params.pv_loop_min_frame_coverage << " not met by " << render_coverage << endl;
			return false;
		}
	}

	return true;
}

CloudICPTargetT::Ptr ObjectModeler::renderBuffersToCloud(RenderBuffers const& render_buffers, cv::Rect const& render_rect)
{
	// assume these are const throughout
	const Eigen::Vector2f render_proj_f(params.camera_focal_x, params.camera_focal_y);
	const Eigen::Vector2f render_proj_c(params.camera_center_x, params.camera_center_y);

	CloudICPTargetT::Ptr result(new CloudICPTargetT);
	result->width = render_rect.width;
	result->height = render_rect.height;
	result->resize(result->width * result->height);

	std::vector<uchar> render_image(render_rect.height * render_rect.width * 4);
	render_buffers.readRenderColorImage(render_image.data());
	std::vector<float> render_normals(render_rect.height * render_rect.width * 4);
	render_buffers.readRenderNormals(render_normals.data());
	std::vector<float> render_points(render_rect.height * render_rect.width * 4);
	render_buffers.readRenderPoints(render_points.data());

	for (int i = 0; i < result->size(); ++i) {
		PointICPTargetT & p = result->at(i);
		p.normal_x = render_normals[4*i];
		p.normal_y = render_normals[4*i+1];
		p.normal_z = render_normals[4*i+2];
		p.x = render_points[4*i];
		p.y = render_points[4*i+1];
		p.z = render_points[4*i+2];
		p.b = render_image[4*i];
		p.g = render_image[4*i+1];
		p.r = render_image[4*i+2];
	}

	return result;
}

bool ObjectModeler::alignWithCombinedOptimizationNew(
	FrameT const& frame,
	RenderBuffers const& render_buffers,
	cv::Rect const& render_rect,
	bool show_debug_images,
	Eigen::Affine3f const& initial_pose,
	Eigen::Affine3f const& initial_relative_pose, 
	Eigen::Affine3f& result_pose)
{
	// assume these are const throughout
	const Eigen::Vector2f render_proj_f(params.camera_focal_x, params.camera_focal_y);
	const Eigen::Vector2f render_proj_c(params.camera_center_x, params.camera_center_y);

	/////////////
	// initialize render
	std::vector<ImageBuffer> rendered_image_channels = getImageChannelsList(render_buffers.getImageBufferColorImage(), render_buffers.getWidth(), render_buffers.getHeight());
	ImageBuffer rendered_image_channels_list = packImageChannelsList(rendered_image_channels, render_buffers.getWidth(), render_buffers.getHeight());

	opencl_optimize_ptr->prepareRenderedAndErrorBuffersWithBuffers(
			render_proj_f[0], render_proj_f[1], render_proj_c[0], render_proj_c[1],
			render_rect.x, render_rect.y, render_rect.width, render_rect.height,
			render_buffers.getBufferPoints(), render_buffers.getBufferNormals(), rendered_image_channels_list.getBuffer());

	///////////
	// also initialize frame
	// frame could be initialized once and first if you want multiple iterations
	const int frame_rows = frame.image_color.rows;
	const int frame_cols = frame.image_color.cols;

	// need uchar4, not uchar3
	ImageBuffer frame_image_bgra(*cl_ptr);
	cv::Mat image_color_bgra;
	cv::cvtColor(frame.image_color, image_color_bgra, CV_BGR2BGRA);
	frame_image_bgra.writeFromBytePointer(image_color_bgra.data, image_color_bgra.rows * image_color_bgra.cols * 4);

	std::vector<ImageBuffer> frame_image_channels = getImageChannelsList(frame_image_bgra, frame_cols, frame_rows);

	// these are passed to optimizer
	ImageBuffer frame_image_buffer(*cl_ptr);
	ImageBuffer frame_gradient_x_buffer(*cl_ptr);
	ImageBuffer frame_gradient_y_buffer(*cl_ptr);

	// gaussian blur
	std::vector<ImageBuffer> split_blurred;
	if (params.color_blur_size > 0) {
		cv::Mat gaussianCoeffs1D = cv::getGaussianKernel(params.color_blur_size, -1, CV_32F);
		for (int i = 0; i < frame_image_channels.size(); ++i) {
			ImageBuffer temp = opencl_images_ptr->convolutionFilterHorizontal(frame_image_channels[i], frame_cols, frame_rows, params.color_blur_size, (float*)gaussianCoeffs1D.data);
			split_blurred.push_back(opencl_images_ptr->convolutionFilterVertical(temp, frame_cols, frame_rows, params.color_blur_size, (float*)gaussianCoeffs1D.data));
		}
	}
	else {
		for (int i = 0; i < frame_image_channels.size(); ++i) {
			split_blurred.push_back(frame_image_channels[i]);
		}
	}

	// sobel filter
	float sobel_smooth[]= {.25f, .5f, .25f};
	float sobel_diff[]= {-.5f, 0.f, .5f};
	std::vector<ImageBuffer> split_gradient_x;
	std::vector<ImageBuffer> split_gradient_y;
	for (int i = 0; i < split_blurred.size(); ++i) {
		ImageBuffer temp = opencl_images_ptr->convolutionFilterVertical(split_blurred[i], frame_cols, frame_rows, 3, sobel_smooth);
		split_gradient_x.push_back(opencl_images_ptr->convolutionFilterHorizontal(temp, frame_cols, frame_rows, 3, sobel_diff));
		temp = opencl_images_ptr->convolutionFilterHorizontal(split_blurred[i], frame_cols, frame_rows, 3, sobel_smooth);
		split_gradient_y.push_back(opencl_images_ptr->convolutionFilterVertical(temp, frame_cols, frame_rows, 3, sobel_diff));
	}

	frame_image_buffer = packImageChannelsList(split_blurred, frame_cols, frame_rows);
	frame_gradient_x_buffer = packImageChannelsList(split_gradient_x, frame_cols, frame_rows);
	frame_gradient_y_buffer = packImageChannelsList(split_gradient_y, frame_cols, frame_rows);

	// reference: old CPU code
#if 0
		cv::Mat frame_image_full_size = cv::Mat(frame_rows, frame_cols, CV_32FC3);
		size_t byte_size = frame_rows * frame_cols * image_channel_count * sizeof(float);
			
		//ImageBuffer frame_image_ycrcb = opencl_images_ptr->extractYCrCbFloat(frame_image_bgra, frame_cols, frame_rows);
		// This, then loop
		ImageBuffer frame_image_full_size_buffer = packImageChannelsList(frame_image_channels, frame_cols, frame_rows);
		frame_image_full_size_buffer.readToBytePointer(frame_image_full_size.data, byte_size); 



		pcl::ScopeTime st ("[TIMING] (new) blur and sobel");
		// blur and compute gradients
		if (params.color_blur_size > 0) {
			cv::GaussianBlur(frame_image_full_size, frame_image_full_size, cv::Size(params.color_blur_size, params.color_blur_size), 0, 0, cv::BORDER_REPLICATE);
		}
		cv::Mat frame_image_gradient_x;
		cv::Mat frame_image_gradient_y;
		cv::Sobel(frame_image_full_size, frame_image_gradient_x, -1, 1, 0, 3, 1.0/8.0, 0, cv::BORDER_REPLICATE);
		cv::Sobel(frame_image_full_size, frame_image_gradient_y, -1, 0, 1, 3, 1.0/8.0, 0, cv::BORDER_REPLICATE);

		// then put the gradient results into the buffers
		frame_gradient_x_buffer.writeFromBytePointer(frame_image_gradient_x.data, byte_size);
		frame_gradient_y_buffer.writeFromBytePointer(frame_image_gradient_y.data, byte_size);
#endif

	const int object_rows = frame.object_rect.height;
	const int object_cols = frame.object_rect.width;
	// but we need a full sized set of points and normals...
	ImageBuffer frame_points_buffer(*cl_ptr);
	ImageBuffer frame_normals_buffer(*cl_ptr);
	if (object_rows != frame_rows || object_cols != frame_cols) {
		// gotta set to nans
		size_t frame_size = frame_rows * frame_cols * 4;
		std::vector<float> nan_vector(frame_size, std::numeric_limits<float>::quiet_NaN());
		frame_points_buffer.writeFromFloatVector(nan_vector);
		frame_normals_buffer.writeFromFloatVector(nan_vector);
		cl::size_t<3> srs_origin;
		srs_origin[0] = 0;
		srs_origin[1] = 0;
		srs_origin[2] = 0;
		cl::size_t<3> dst_origin;
		dst_origin[0] = frame.object_rect.x * 4 * sizeof(float);
		dst_origin[1] = frame.object_rect.y; // * 4 * sizeof(float);
		dst_origin[2] = 0;
		cl::size_t<3> region;
		region[0] = frame.object_rect.width * 4 * sizeof(float);
		region[1] = frame.object_rect.height; // * 4 * sizeof(float);
		region[2] = 1;
		size_t src_row_pitch = object_cols * 4 * sizeof(float);
		size_t dst_row_pitch = frame_cols * 4 * sizeof(float);

		try {
			cl_ptr->queue.enqueueCopyBufferRect(frame.image_buffer_points_ptr->getBuffer(), frame_points_buffer.getBuffer(), srs_origin, dst_origin, region, src_row_pitch, 0, dst_row_pitch, 0);
			cl_ptr->queue.enqueueCopyBufferRect(frame.image_buffer_normals_ptr->getBuffer(), frame_normals_buffer.getBuffer(), srs_origin, dst_origin, region, src_row_pitch, 0, dst_row_pitch, 0);
		}
		catch (cl::Error er) {
			printf("cl::Error: %s\n", oclErrorString(er.err()));
			printf("copy to full size points and normals\n");
			throw er;
		}
	}
	else {
		frame_points_buffer = *frame.image_buffer_points_ptr;
		frame_normals_buffer = *frame.image_buffer_normals_ptr;
	}

	opencl_optimize_ptr->prepareFrameBuffersWithBuffers(
		params.camera_focal_x, params.camera_focal_y, params.camera_center_x, params.camera_center_y,
		frame.object_rect.x, frame.object_rect.y, frame.object_rect.width, frame.object_rect.height,
		frame_cols, frame_rows,
		frame_points_buffer.getBuffer(), frame_normals_buffer.getBuffer(), frame_image_buffer.getBuffer(), frame_gradient_x_buffer.getBuffer(), frame_gradient_y_buffer.getBuffer());


	////////////////
	// run the optimization
	static pcl::WarpPointRigid6D<PointT, PointT> warp_point;
	Eigen::VectorXf x_result(6);
	x_result.setZero();
	int iterations = 0;
	for ( ; iterations < params.combined_gauss_newton_max_iterations; iterations++) {
		Eigen::Matrix<float,6,6> LHS;
		Eigen::Matrix<float,6,1> RHS;

		warp_point.setParam (x_result);
		Eigen::Affine3f x_transform(warp_point.getTransform());

		// new: allow an initialization of the relative pose
		x_transform = x_transform * initial_relative_pose;

		std::vector<float> error_vector;
		float* error_vector_ptr = NULL;
		std::vector<float> weight_vector;
		float* weight_vector_ptr = NULL;
		if (show_debug_images) {
			error_vector.resize(opencl_optimize_ptr->getErrorVectorSize());
			error_vector_ptr = error_vector.data();
			weight_vector.resize(opencl_optimize_ptr->getErrorVectorSize());
			weight_vector_ptr = weight_vector.data();
		}

		// params for optimization
		const float optimize_max_distance = params.icp_max_distance;
		const float optimize_min_normal_dot = cos(params.icp_normals_angle * M_PI / 180.0);
		const float optimize_weight_icp = std::max(params.combined_weight_icp_points, 0.f);
		const float optimize_weight_color = std::max(params.combined_weight_color, 0.f);

		// actually optimize
		opencl_optimize_ptr->computeErrorAndGradient(
			optimize_max_distance, optimize_min_normal_dot,
			optimize_weight_icp, optimize_weight_color,
			x_transform,
			LHS, RHS, error_vector_ptr, NULL, weight_vector_ptr);

		if (params.combined_verbose) {
			cout << "LHS:\n" << LHS << endl;
			cout << "RHS:\n" << RHS.transpose() << endl;
		}

		Eigen::VectorXf x_delta = LHS.ldlt().solve(-RHS);
		x_result += x_delta;

		if (params.combined_verbose) {
			cout << "x_delta:\n" << x_delta.transpose() << endl;
			cout << "x_result:\n" << x_result.transpose() << endl;
		}

		if (show_debug_images) {
			// pull images out of ImageBuffers
			// TODO: Not this if displaying weights instead of useless target image
			std::vector<cv::Mat> frame_images;
			for (int i = 0; i < frame_image_channels.size(); ++i) {
				frame_images.push_back(cv::Mat(frame_rows, frame_cols, CV_32F));
				frame_image_channels[i].readToBytePointer(frame_images.back().data, frame_rows * frame_cols * sizeof(float));
			}
			combinedDebugImagesNew(frame, x_transform, getImageChannelCount(), error_vector, weight_vector, render_rect, frame_images, frame.object_rect, render_buffers, render_proj_f, render_proj_c);
		}


		/////////// continue?
		if (params.combined_gauss_newton_min_delta_to_continue > 0) {
			// get the max(abs(x_delta))
			float max_component = x_delta.array().abs().maxCoeff();
			if (max_component < params.combined_gauss_newton_min_delta_to_continue) break;
		}
	}

	rs_gn_iterations.push(iterations);
	cout << "Gauss-Newton Iterations: " << rs_gn_iterations.summary() << endl;

	warp_point.setParam(x_result);
	Eigen::Affine3f pose_correction = Eigen::Affine3f(warp_point.getTransform());
	result_pose = pose_correction * initial_relative_pose * initial_pose;

	return true;
}

// This has some ideas in it...don't delete
#if 0
bool ObjectModeler::alignWithCombinedOptimization(
	const FrameT& frame,
	bool is_loop_closure,
	const Eigen::Affine3f& initial_pose,
	Eigen::Affine3f& result_pose,
	CloudICPTargetT::Ptr& last_render_cloud,
	Eigen::Affine3f& pose_for_last_render_cloud,
	cv::Mat& which_segment)
{
	pcl::ScopeTime st("[TIMING] alignWithCombinedOptimization");
	Eigen::Affine3f current_pose = initial_pose;
	globalStopWatchMark("beginning of alignWithCombinedOptimization");

	int iteration = 0;
	int min_iterations = 0; // shouldn't be needed
	int max_iterations = max(params.icp_max_iterations, params.combined_octaves);
	int current_octave = params.combined_octaves - 1;

	float last_error_icp = 0;
	float last_error_icp_max = 0;
	int last_error_icp_count = 0;
	float last_error_color = 0;
	float last_error_color_max = 0;
	int last_error_color_count = 0;
	float last_error_total = 0;
	float last_error_according_to_optimizer = 0;
	int last_render_point_count = 0;
	int image_channel_count = 0; // set to actual value from functor later
	float last_ev_min_before = 0;
	float last_ev_min_after = 0;
	float last_rank_after = 0; // int?

	CloudT::Ptr icp_debug_cloud(new CloudT);
	Eigen::Affine3f estimate_to_apply_to_current_pose = Eigen::Affine3f::Identity();

	while(++iteration <= max_iterations) {
		pcl::StopWatch sw;
		globalStopWatchMark("before render for icp");

		cv::Rect render_rect = frame.object_rect;
		Eigen::Vector2f render_proj_f(params.camera_focal_x, params.camera_focal_y);
		Eigen::Vector2f render_proj_c(params.camera_center_x, params.camera_center_y);

		// overwrite scale if combined_octaves > 1
		float scale = params.combined_render_scale;
		if (params.combined_octaves > 1) {
			scale = 1.0 / (1 << current_octave);
		}

		render_rect.x *= scale;
		render_rect.y *= scale;
		render_rect.width *= scale;
		render_rect.height *= scale;
		render_proj_f *= scale;
		render_proj_c *= scale;

		CloudT::Ptr rendered_cloud(new CloudT);
		pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
		RenderBuffers render_buffers(*cl_ptr);
		if (params.use_patch_volumes) {
			// Somewhat dangerously don't lock here because I know they won't be changed while this is running
			// really lock should be in renderPatchVolumes, dingus
			// can't lock here because this gets called in updatedPatchVolumes during loop closure...
			//boost::mutex::scoped_lock lock(mutex_volumes);
			int pv_min_age = is_loop_closure ? (params.pv_max_age + 1) : -1;
			int pv_max_age = is_loop_closure ? -1 : params.pv_max_age;
			float pv_max_normal_angle = is_loop_closure ? params.pv_loop_max_normal_angle : -1;

			renderPatchVolumes(current_pose, render_proj_f, render_proj_c, render_rect, pv_min_age, pv_max_age, pv_max_normal_angle, *rendered_cloud, *normal_cloud, which_segment, render_buffers);
			if (params.pv_debug_show_render_for_alignment) {
				cv::Mat colors = cloudToImage(*rendered_cloud, cv::Vec3b(0,0,255));
				cv::Mat normals = normalCloudToImage(*normal_cloud, light_direction, cv::Vec3b(0,0,255));
				showInWindow("pv_debug_show_render_for_alignment", create1x2(colors, normals));
				cout << "pauseAndShowImages for pv_debug_show_render_for_alignment" << endl;
				pauseAndShowImages();
			}
		}
		else {
			renderVolumeWithOpenCL(current_pose, render_proj_f, render_proj_c, render_rect, *rendered_cloud, *normal_cloud, render_buffers);
		}

		////////////////////////////////////
		// pack rendered cloud + normals into one PointICPTargetT
		// arguably render should simply return this cloud type
		// or perhaps render should just return images!!
		CloudICPTargetT::Ptr rendered_cloud_with_normals(new CloudICPTargetT);
		last_render_point_count = 0;
		{
			pcl::ScopeTime st("[TIMING] Create rendered_cloud_with_normals");
			if (rendered_cloud->size() != normal_cloud->size()) {
				throw std::runtime_error("rendered_cloud->size() != normal_cloud->size()");
			}
			rendered_cloud_with_normals->points.reserve(rendered_cloud->size());
			for (size_t i = 0; i < rendered_cloud->size(); i++) {
				const PointT& p_source = rendered_cloud->points[i];
				const pcl::Normal& n_source = normal_cloud->points[i];
				PointICPTargetT p_target;
				p_target.getVector4fMap() = p_source.getVector4fMap();
				p_target.getNormalVector4fMap() = n_source.getNormalVector4fMap();
				p_target.r = p_source.r;
				p_target.g = p_source.g;
				p_target.b = p_source.b;
				rendered_cloud_with_normals->points.push_back(p_target);

				// count valid
				if (pcl_isfinite(p_target.z) && pcl_isfinite(p_target.normal_z)) last_render_point_count++;
			}
			rendered_cloud_with_normals->width = rendered_cloud->width;
			rendered_cloud_with_normals->height = rendered_cloud->height;
			rendered_cloud_with_normals->is_dense = false;
		}
		last_render_cloud = rendered_cloud_with_normals;
		pose_for_last_render_cloud = current_pose;

		///////////////////////////
		// debug the iteration
		// the last rendered cloud (in green):
		{
			pcl::ScopeTime st("[TIMING] create icp_debug_cloud");
			icp_debug_cloud->reserve(rendered_cloud_with_normals->size());
			icp_debug_cloud->clear();
			for (int i = 0; i < rendered_cloud_with_normals->size(); i++) {
				const PointICPTargetT& p = rendered_cloud_with_normals->points[i];
				if (pcl_isfinite(p.z)) {
					PointT new_p;
					new_p.getVector4fMap() = p.getVector4fMap();
					// alternate colors between green and different green and green based on input_frame_counter (see new frames better)
					if (input_frame_counter % 2 == 0) {
						new_p.r = 0;
						new_p.g = 255;
						new_p.b = 0;
					}
					else {
						new_p.r = 0;
						new_p.g = 180;
						new_p.b = 0;
					}
					icp_debug_cloud->points.push_back(new_p);
				}
			}
			icp_debug_cloud->width = icp_debug_cloud->size();
			icp_debug_cloud->height = 1;
			icp_debug_cloud->is_dense = true;
			tc_icp_debug_points.setCloud(icp_debug_cloud);
		}
		//////////////////////////// end debug

		cout << "last_render_point_count: " << last_render_point_count << endl;
		if (params.combined_min_rendered_point_count > 0 && last_render_point_count < params.combined_min_rendered_point_count) {
			cout << "min_rendered_point_count " << params.combined_min_rendered_point_count << " not met by last_render_point_count " << last_render_point_count << endl;
			return false;
		}

		globalStopWatchMark("after render for icp");

		// to be filled in with optimized result:
		Eigen::Matrix4f transformation_matrix;

		//typedef double ScalarTypeForLM;
		typedef float ScalarTypeForLM;
		typedef ICPCombinedFunctor<ScalarTypeForLM> ICPCombinedFunctorT;
		typedef Eigen::Matrix<ScalarTypeForLM, Eigen::Dynamic, 1> ICPCombinedFunctorVectorT;

		static const unsigned int n_unknowns = 6;
		ICPCombinedFunctorVectorT lm_x(n_unknowns);
		lm_x.setZero (); // this is initializing to the identity

		// run the optimization
		// Note that we compute the whole frame pyramid every constructor
		// consider moving functor initialization outside of loop...also allows a single init of the frame part of the functor
		ICPCombinedFunctorT icp_combined_functor(params, 
			g2o_stereo_projector, current_pose,
			frame, opencl_optimize_ptr.get(), opencl_images_ptr.get());
		image_channel_count = icp_combined_functor.getImageChannelCount();

		// separate initFrame allows you to select the octave
		{
			pcl::ScopeTime st("[TIMING] icp_combined_functor.initFrame");
			icp_combined_functor.initFrame(current_octave);
		}

		// separating initRender allows for updating the render outside the functor
		{
			pcl::ScopeTime st("[TIMING] icp_combined_functor.initRender");
			icp_combined_functor.initRender(render_proj_f, render_proj_c, render_rect, render_buffers);
		}

		globalStopWatchMark("after prepare functor");

		if (params.combined_gauss_newton) {
			pcl::ScopeTime st("[TIMING] combined_gauss_newton");
			ICPCombinedFunctorVectorT gn_x_result (n_unknowns);
			int gn_iterations;
			Eigen::Matrix<ICPCombinedFunctorT::Scalar, Eigen::Dynamic, 1> gn_error_vector;
			icp_combined_functor.solveGaussNewton(lm_x, gn_x_result, gn_iterations, gn_error_vector);
			lm_x = gn_x_result;
			last_error_according_to_optimizer = gn_error_vector.norm();
			rs_lm_iterations.push(gn_iterations);
		}
		else if (params.combined_gauss_newton_gpu_full) {
			pcl::ScopeTime st("[TIMING] combined_gauss_newton_gpu_full");
			ICPCombinedFunctorVectorT gn_x_result (n_unknowns);
			int gn_iterations;
			icp_combined_functor.solveGaussNewtonGPUFull(lm_x, gn_x_result, gn_iterations);
			lm_x = gn_x_result;
			last_error_according_to_optimizer = 0; // yeah...figure this out...
			rs_lm_iterations.push(gn_iterations);
		}
		else {
			pcl::ScopeTime st("[TIMING] Eigen::LevenbergMarquardt");
			int lm_info = -100;
			Eigen::LevenbergMarquardt<ICPCombinedFunctorT, ScalarTypeForLM> lm (icp_combined_functor);
			lm_info = lm.minimize (lm_x);
			last_error_according_to_optimizer = lm.fvec.norm();
			// we can get this from functor:
			//last_error_nonzero_entries = (lm.fvec.array().abs() > 0).cast<int>().sum();
			rs_lm_iterations.push(lm.iter);
		}
		cout << "f() timing (This ICP iteration) (ms): " << endl << icp_combined_functor.rs_f.summary() << endl;
		cout << "df() timing (This ICP iteration) (ms): " << endl << icp_combined_functor.rs_df.summary() << endl;
		cout << "LM iterations (count): " << endl << rs_lm_iterations.summary() << endl;

		globalStopWatchMark("optimization");

		// debug
		if (params.combined_debug_normal_eq) {
			//Eigen::IOFormat io_full(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

			// to compare against:
			ICPCombinedFunctorT::JacobianType J(icp_combined_functor.values(), 6);
			icp_combined_functor.df(lm_x, J);
			typedef Eigen::Matrix<ScalarTypeForLM, Eigen::Dynamic, Eigen::Dynamic> GeneralJType;
			GeneralJType JTJ = J.transpose()*J;
			ICPCombinedFunctorT::ValueType V(icp_combined_functor.values());
			icp_combined_functor(lm_x, V);
			Eigen::Matrix<float, 6, 1> JTV = J.transpose()*V;

			cout << "JtJ CPU:\n" << JTJ << endl;
			cout << "JtV CPU:\n" << JTV << endl;

			// even newer code
			{
				pcl::ScopeTime st("computeErrorAndGradient()");
				Eigen::Affine3f t(icp_combined_functor.xToMatrix4f(lm_x));
				Eigen::Matrix<float,6,6> LHS;
				Eigen::Matrix<float,6,1> RHS;

				// so much debug::::
				std::vector<float> error_matrix(opencl_optimize_ptr->getErrorMatrixSize(), 0);
				std::vector<float> error_vector(opencl_optimize_ptr->getErrorVectorSize(), 0);

				//////////////
				opencl_optimize_ptr->computeErrorAndGradient(t, LHS, RHS, error_vector.data(), error_matrix.data());
				//////////////

				int error_channels = icp_combined_functor.getErrorChannelCount();
				int error_point_count = icp_combined_functor.errorPoints();

				///////////
				//////////// error matrix
				ICPCombinedFunctorT::ValueType Vtest(icp_combined_functor.values());
				for (int c = 0; c < error_channels; ++c) {
					for (int row = 0; row < error_point_count; ++row) {
						Vtest(c*error_point_count + row) = 
							error_vector[c*error_point_count + row];
					}
				}

				////////// J matrix
				// copied out of the functor:
				// In the new version, it's by channel, col, row
				// and error is packed channel, row
				ICPCombinedFunctorT::JacobianType Jtest(icp_combined_functor.values(), 6);
				for (int c = 0; c < error_channels; ++c) {
					for (int row = 0; row < error_point_count; ++row) {
						for (int col = 0; col < 6; ++col) {
							Jtest(c*error_point_count+row, col) = 
								error_matrix[c*error_point_count*6 + col*error_point_count + row];
						}
					}
				}
				Eigen::Matrix<float,6,6> view = Eigen::Matrix<float,6,6>::Zero();
				//view = (Jtest.transpose()*Jtest).triangularView<Eigen::Upper>();
				view = (Jtest.transpose()*Jtest);
				cout << "LHSTest:\n" << view << endl;

				cout << "LHS:\n" << LHS << endl;

				//////////
				// Now check RHS just for fun:
				Eigen::Matrix<float,6,1> RHSTest = Jtest.transpose()*Vtest;
				cout << "RHSTest:\n" << RHSTest << endl;

				cout << "RHS:\n" << RHS << endl;
			}

			// assume you want pausing here
			cout << "combined_debug_normal_eq waitKey()..." << endl;
			cv::waitKey();
		}


		// intent: get the min eigenvalue before and after all optimizations
		// As this is non-trivially slow on my new stupid machine, only compute if needed
		if (params.error_min_eigenvalue > 0) {
			if (current_octave == 0) {
				pcl::ScopeTime st("[TIMING] Eigenvalues of JTJ (at 0 vec)");
				ICPCombinedFunctorVectorT lm_zero(n_unknowns);
				lm_zero.setZero ();
				ICPCombinedFunctorT::JacobianType J(icp_combined_functor.values(), 6);
				icp_combined_functor.df(lm_zero, J);
				typedef Eigen::Matrix<ScalarTypeForLM, Eigen::Dynamic, Eigen::Dynamic> GeneralJType;
				GeneralJType JTJ = J.transpose()*J;
				Eigen::SelfAdjointEigenSolver<GeneralJType> es(JTJ);
				float min_eigenvalue = es.eigenvalues().minCoeff();
				last_ev_min_before = min_eigenvalue;
			}

			if (current_octave == 0) {
				pcl::ScopeTime st("[TIMING] Eigenvalues of JTJ");
				ICPCombinedFunctorT::JacobianType J(icp_combined_functor.values(), 6);
				icp_combined_functor.df(lm_x, J);
				typedef Eigen::Matrix<ScalarTypeForLM, Eigen::Dynamic, Eigen::Dynamic> GeneralJType;
				GeneralJType JTJ = J.transpose()*J;
				Eigen::SelfAdjointEigenSolver<GeneralJType> es(JTJ);
				float min_eigenvalue = es.eigenvalues().minCoeff();
				last_ev_min_after = min_eigenvalue;
			}

			// also consider better ways to get the nullspace of JTJ
			// god damn it I wish I understood linear algebra...I'm sure I can get this out of the eigenvalue thing as well
			if (current_octave == 0) {
				pcl::ScopeTime st("[TIMING] LU decomposition of JTJ");
				ICPCombinedFunctorT::JacobianType J(icp_combined_functor.values(), 6);
				icp_combined_functor.df(lm_x, J);
				typedef Eigen::Matrix<ScalarTypeForLM, Eigen::Dynamic, Eigen::Dynamic> GeneralJType;
				GeneralJType JTJ = J.transpose()*J;
				Eigen::FullPivLU<GeneralJType> lu_decomp(JTJ);
				if (params.error_rank_threshold > 0) {
					lu_decomp.setThreshold(params.error_rank_threshold);
				}
				int rank = lu_decomp.rank();
				last_rank_after = rank;
			}
		}

		last_error_icp = icp_combined_functor.last_error_icp_;
		last_error_color = icp_combined_functor.last_error_color_;
		last_error_total = icp_combined_functor.last_error_total_;
		last_error_icp_max = icp_combined_functor.last_error_icp_max_;
		last_error_color_max = icp_combined_functor.last_error_color_max_;
		last_error_icp_count = icp_combined_functor.last_error_icp_count_;
		last_error_color_count = icp_combined_functor.last_error_color_count_;

		transformation_matrix = icp_combined_functor.xToMatrix4f(lm_x);


		////////////////////////////
		// This is common whichever way you got the transformation_matrix
		Eigen::Affine3f estimate_as_affine(transformation_matrix);
		// get change in translation and rotation:
		float translation_norm = estimate_as_affine.translation().norm();
		Eigen::AngleAxisf get_angle_amount(estimate_as_affine.rotation());
		float rotation_angle = get_angle_amount.angle();
		cout << "For iteration: " << iteration << endl;
		cout << "Translation distance: " << translation_norm << endl;
		cout << "Rotation angle: " << rotation_angle << endl;

		// updated current pose:
		estimate_to_apply_to_current_pose = estimate_as_affine;
		current_pose = estimate_to_apply_to_current_pose * current_pose;

		rs_time_per_icp_iteration.push(sw.getTime());

		// save the debug images from the functor
		if (params.save_cdi_images) {
			cout << "Saving cdi images..." << endl;
			static int debug_image_counter = 0;
			for (int i = 0; i < icp_combined_functor.combined_debug_image_v.size(); ++i) {
				savePNGWithIndex("combined-debug-image", icp_combined_functor.combined_debug_image_v.at(i), debug_image_counter++);
			}
			cout << "...done" << endl;
		}

		// some arbitrary cutoffs
		if (iteration >= min_iterations) {
			if (translation_norm < params.icp_min_translation_to_continue && rotation_angle < params.icp_min_rotation_to_continue) break;
		}

		--current_octave;
		if (current_octave < 0) current_octave = 0;

		if (params.combined_debug_pause_after_icp_iteration) {
			cout << "combined_debug_pause_after_icp_iteration waitKey()..." << endl;
			cv::waitKey();
		}
	}
	rs_icp_iterations.push(min(iteration, max_iterations));

	result_pose = current_pose;
	bool success = true;

	// always have timing info
	cout << "ICP iterations (count):\n" << rs_icp_iterations.summary() << endl;
	cout << "ICP time per iteration (ms):\n" << rs_time_per_icp_iteration.summary() << endl;

	// store error information and perhaps act on it:
	{
		//pcl::ScopeTime st("tables of values"); // 0ms
		cout << "Final error icp: " << last_error_icp << endl;
		cout << "Final error color: " << last_error_color << endl;
		cout << "Final error total: " << last_error_total << endl;
		cout << "Final error according to optimizer: " << last_error_according_to_optimizer << endl;

		float normalized_icp_error = last_error_icp / last_error_icp_count;
		float normalized_color_error = last_error_color / last_error_color_count;
		int total_nonzero_entries = last_error_icp_count + last_error_color_count;
		float normalized_error = last_error_according_to_optimizer / total_nonzero_entries;

		tables_of_values["error-total"].push_back(last_error_according_to_optimizer);
		tables_of_values["render-count"].push_back(last_render_point_count);
		tables_of_values["nonzero-count"].push_back(total_nonzero_entries);
		tables_of_values["nonzero-icp"].push_back(last_error_icp_count);
		tables_of_values["nonzero-color"].push_back(last_error_color_count);
		tables_of_values["error-norm"].push_back(normalized_error);

		// compute the outlier fraction for ICP and color
		float inlier_fraction_icp = (float) last_error_icp_count / (float) last_render_point_count;
		float inlier_fraction_color = (float) last_error_color_count / (float) (image_channel_count * last_render_point_count);
		tables_of_values["inlier-icp"].push_back(inlier_fraction_icp);
		tables_of_values["inlier-color"].push_back(inlier_fraction_color);

		// also error by types
		tables_of_values["error-icp"].push_back(last_error_icp);
		tables_of_values["error-color"].push_back(last_error_color);
		tables_of_values["error-norm-icp"].push_back(normalized_icp_error);
		tables_of_values["error-norm-color"].push_back(normalized_color_error);
		tables_of_values["max-icp"].push_back(last_error_icp_max);
		tables_of_values["max-color"].push_back(last_error_color_max);

		// also transformation change
		Eigen::Affine3f overall_relative_transform_from_icp = result_pose * initial_pose.inverse();
		float translation_norm = overall_relative_transform_from_icp.translation().norm();
		Eigen::AngleAxisf get_angle_amount(overall_relative_transform_from_icp.rotation());
		float rotation_angle = get_angle_amount.angle();
		tables_of_values["pose-t"].push_back(translation_norm);
		tables_of_values["pose-r"].push_back(rotation_angle);

		// also eigenvalues
		tables_of_values["eigen-min-before"].push_back(last_ev_min_before);
		tables_of_values["eigen-min-after"].push_back(last_ev_min_after);

		// and nullspace attempts
		tables_of_values["rank-after"].push_back(last_rank_after);

		// attempt to use error based on "analyzing" the data
		// instead of using the tables, i'll commit to a new variable
		//float color_error_diff = getLastDelta("error-color");
		// slightly different because previous_error_color only on success
		float color_error_diff = last_error_color - previous_error_color;
		float color_error_diff_relative = max<float>(0, color_error_diff / last_error_color);
		tables_of_values["color-error-diff-relative"].push_back(color_error_diff_relative);
		// also track the previous_error_color, which should remain static during failures..
		tables_of_values["previous-error-color"].push_back(previous_error_color);

		// actually act on the error!
		const static std::string fail_prefix = "\n-- FAIL: ";
		if (params.error_max_t > 0 && translation_norm > params.error_max_t) {
			cout << fail_prefix << "translation_norm: " << translation_norm << endl;
			success = false;
		}
		if (params.error_max_r > 0 && rotation_angle > params.error_max_r) {
			cout << fail_prefix << "rotation_angle: " << rotation_angle << endl;
			success = false;
		}
		if (output_frame_counter >= params.error_min_output_frames && params.error_change > 0 && color_error_diff_relative > params.error_change) {
			cout << fail_prefix << "color_error_diff_relative: " << color_error_diff_relative << endl;
			success = false;
		}
		if (output_frame_counter >= params.error_min_output_frames && params.error_min_inlier_fraction_icp > 0 && inlier_fraction_icp < params.error_min_inlier_fraction_icp) {
			cout << fail_prefix << "inlier_fraction_icp: " << inlier_fraction_icp << endl;
			success = false;
		}
		if (output_frame_counter >= params.error_min_output_frames && params.error_min_inlier_fraction_color > 0 && inlier_fraction_color < params.error_min_inlier_fraction_color) {
			cout << fail_prefix << "inlier_fraction_color: " << inlier_fraction_color << endl;
			success = false;
		}
		if (output_frame_counter >= params.error_min_output_frames && last_render_point_count < params.error_min_rendered_points) {
			cout << fail_prefix << "last_render_point_count: " << last_render_point_count << endl;
			success = false;
		}
		if (params.error_min_eigenvalue > 0 && last_ev_min_after < params.error_min_eigenvalue) {
			cout << fail_prefix << "last_ev_min_after: " << last_ev_min_after << endl;
			success = false;
		}
		if (params.error_use_rank && last_rank_after < 6) {
			cout << (boost::format(fail_prefix + "error_use_rank and last_rank_after: %d (threshold: %f)") % last_rank_after % params.error_rank_threshold).str() << endl;
			success = false;
		}

		// only update the previous error if this frame was a success
		if (success) previous_error_color = last_error_color;

		// track success
		tables_of_values["icp-success"].push_back((float)success);	
	}

	// show the final pose in red:
	{
		pcl::transformPointCloud(*icp_debug_cloud, *icp_debug_cloud, estimate_to_apply_to_current_pose);
		for (size_t i = 0; i < icp_debug_cloud->size(); i++) {
			PointT& p = icp_debug_cloud->points[i];
			if (success) {
				// purple good
				p.r = 255;
				p.g = 0;
				p.b = 255;
			}
			else {
				// red bad
				p.r = 255;
				p.g = 0;
				p.b = 0;
			}
		}
		tc_icp_debug_points.setCloud(icp_debug_cloud);
	}

	globalStopWatchMark("after optimization stuff");

	if (params.combined_debug_pause_after_icp_all) {
		cout << "combined_debug_pause_after_icp_all pauseAndShowImages()..." << endl;
		pauseAndShowImages();
	}

	return success;
}
#endif

pcl::PointCloud<pcl::Normal>::Ptr ObjectModeler::computeNormalsCrossProduct(const CloudT::ConstPtr& cloud)
{
	const float max_depth_difference = 0.1; // used to be a param, but that just confused me
	static const pcl::Normal nan_normal(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
	const float distance_threshold_squared = max_depth_difference * max_depth_difference;

	pcl::PointCloud<pcl::Normal>::Ptr result(new pcl::PointCloud<pcl::Normal>);
	result->width = cloud->width;
	result->height = cloud->height;
	result->is_dense = cloud->is_dense;
	result->points.assign(result->width * result->height, nan_normal);
	for (int row = 1; row < cloud->height-1; row++) {
		for (int col = 1; col < cloud->width-1; col++) {
			const PointT& p = cloud->at(col, row);
			if (pcl_isnan(p.z)) continue;
			const PointT& px0 = cloud->at(col-1, row);
			if (pcl_isnan(px0.z)) continue;
			const PointT& px1 = cloud->at(col+1, row);
			if (pcl_isnan(px1.z)) continue;
			const PointT& py0 = cloud->at(col, row-1);
			if (pcl_isnan(py0.z)) continue;
			const PointT& py1 = cloud->at(col, row+1);
			if (pcl_isnan(py1.z)) continue;

			// compute distance to all surrounding points, and if any are over the threhold, skip
			if ( (p.getVector3fMap() - px0.getVector3fMap()).squaredNorm() > distance_threshold_squared ||
				(p.getVector3fMap() - px1.getVector3fMap()).squaredNorm() > distance_threshold_squared ||
				(p.getVector3fMap() - py0.getVector3fMap()).squaredNorm() > distance_threshold_squared ||
				(p.getVector3fMap() - py1.getVector3fMap()).squaredNorm() > distance_threshold_squared) continue;

			// compute the cross product of the vectors:
			Eigen::Vector3f normal = ( (py1.getVector3fMap()-py0.getVector3fMap()).cross(px1.getVector3fMap()-px0.getVector3fMap()) ).normalized();
			result->at(col, row).getNormalVector3fMap() = normal;
		}
	}

	return result;
}

pcl::PointCloud<pcl::Normal>::Ptr ObjectModeler::smoothNormals(const CloudT::ConstPtr& cloud, const pcl::PointCloud<pcl::Normal>::ConstPtr& input_normals)
{
	const float max_depth_difference = 0.1; // used to be a param, but that just confused me
	static const pcl::Normal nan_normal(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
	const float distance_threshold_squared = max_depth_difference * max_depth_difference;

	pcl::PointCloud<pcl::Normal>::Ptr output_normals(new pcl::PointCloud<pcl::Normal>());
	output_normals->points.assign(input_normals->size(), nan_normal);
	output_normals->width = input_normals->width;
	output_normals->height = input_normals->height;
	output_normals->is_dense = input_normals->is_dense;

	for (int row = 0; row < input_normals->height; row++) {
		for (int col = 0; col < input_normals->width; col++) {
			const PointT& p_center = cloud->at(col, row);
			if (pcl_isnan(p_center.z)) continue;
			Eigen::Vector3f normal_sum(0,0,0);
			bool any_valid_normals = false;
			for (int row_d = -1; row_d <= 1; row_d++) {
				for (int col_d = -1; col_d <= 1; col_d++) {
					int row_this = row + row_d;
					int col_this = col + col_d;

					// check out of bounds...could avoid by only smoothing between 1 and n-1?
					if (row_this < 0 || row_this >= input_normals->height || col_this < 0 || col_this >= input_normals->width) continue;

					// new: only smooth when within distance threshold
					const PointT& p_other = cloud->at(col_this, row_this);
					if (pcl_isnan(p_other.z)) continue;
					if ((p_center.getVector3fMap() - p_other.getVector3fMap()).squaredNorm() > distance_threshold_squared) continue;

					// use valid normals:
					const pcl::Normal& n = input_normals->at(col_this, row_this);
					if (pcl_isnan(n.normal_z)) continue;
					normal_sum += n.getNormalVector3fMap();
					any_valid_normals = true;
				}
			}
			// keep the initial NaN value if no valid normals surround this point
			if (any_valid_normals) {
				output_normals->at(col, row).getNormalVector3fMap() = normal_sum.normalized();
			}
		}
	}

	return output_normals;
}

pcl::PointCloud<pcl::Normal>::Ptr ObjectModeler::computeNormals(const CloudT::ConstPtr& cloud)
{
	pcl::PointCloud<pcl::Normal>::Ptr result = computeNormalsCrossProduct(cloud);

	for (int smooth = 0; smooth < params.normals_smooth_iterations; ++smooth) {
		result = smoothNormals(cloud, result);
	}

	return result;
}

cv::Mat ObjectModeler::getMaskOfValidNormals(const pcl::PointCloud<pcl::Normal>& normal_cloud)
{
	cv::Mat result(normal_cloud.height, normal_cloud.width, CV_8UC1, cv::Scalar(0));
	for (int row = 0; row < normal_cloud.height; ++row) {
		for (int col = 0; col < normal_cloud.width; ++col) {
			if (pcl_isfinite(normal_cloud.at(col, row).normal_z)) result.at<unsigned char>(row, col) = 255;
		}
	}
	return result;
}

void ObjectModeler::addObjectFeaturesToFrame(FrameT& frame)
{
	boost::scoped_ptr<cv::FeatureDetector> detector_ptr;
	boost::scoped_ptr<cv::DescriptorExtractor> extractor_ptr;

	if (feature_type == FEATURE_TYPE_FAST) {
		// FAST / BRIEF
		int fastThreshold = 20; // default is 10
		detector_ptr.reset(new cv::FastFeatureDetector(fastThreshold));
		int briefBytes = 32; // 16, 32, 64 are only ones (32 is default)
		extractor_ptr.reset(new cv::BriefDescriptorExtractor(briefBytes));
	}
	else if (feature_type == FEATURE_TYPE_SURF) {
		// SURF
		int minHessian = 300; // 400 default 
		int nOctaves = 2; // 3 default
		int nOctaveLayers = 4; // 4 default
		bool upright = false; // false default
		detector_ptr.reset(new cv::SurfFeatureDetector(minHessian, nOctaves, nOctaveLayers, upright));
		extractor_ptr.reset(new cv::SurfDescriptorExtractor());
	}
	else if (feature_type == FEATURE_TYPE_ORB) {
		// there are probably other parameters
		int nfeatures=500;
		detector_ptr.reset(new cv::OrbFeatureDetector(nfeatures));
		extractor_ptr.reset(new cv::OrbDescriptorExtractor()); // patch size??
	}
	else {
		throw std::runtime_error("unknown feature type");
	}

	// extract features for the masked region of frame
	cv::Mat image_for_features = frame.image_color;
	cv::Mat mask_for_features;
	if (params.mask_object) {
		int erode_iterations = 3;
		cv::erode(frame.object_mask, mask_for_features, cv::noArray(), cv::Point(-1,-1), erode_iterations);
	}
	else {
		mask_for_features = frame.object_mask;
	}

	// detect keypoints and extract descriptors
	// note that extractor can eliminate some keypoints
	detector_ptr->detect( image_for_features, frame.object_kp.keypoints, mask_for_features);
	extractor_ptr->compute( image_for_features, frame.object_kp.keypoints, frame.object_kp.descriptors);
	frame.object_kp.inlier_count.assign(frame.object_kp.keypoints.size(), 0);

	// add the corresponding 3D points
	size_t num_keypoints = frame.object_kp.keypoints.size();
	frame.object_kp.keypoint_cloud.reset(new FrameT::KeypointCloudT);
	frame.object_kp.keypoint_cloud->resize(num_keypoints);
	// for later filtering of invalid cloud points
	std::vector<bool> keypoint_filter_vec(num_keypoints, true);
	bool need_to_filter_keypoints = false;
	for (size_t i = 0; i < num_keypoints; i++) {
		const cv::KeyPoint& cv_kp = frame.object_kp.keypoints[i];
		int cloud_row = (int) (cv_kp.pt.y + 0.5);
		int cloud_col = (int) (cv_kp.pt.x + 0.5);
		int cloud_index = cloud_row * frame.cloud_ptr->width + cloud_col;
		const PointT& p = frame.cloud_ptr->points[cloud_index];
	
		// This happens for SURF (haven't seen it for FAST as long as there's no dilation of the mask)
		if (pcl_isnan(p.z)) {
			need_to_filter_keypoints = true;
			keypoint_filter_vec[i] = false;
		}

		// keypoint points can be different type...currently only need xyz
		frame.object_kp.keypoint_cloud->points[i].getVector4fMap() = p.getVector4fMap();
	}

	// now filter out invalid 3d points
	if (need_to_filter_keypoints) {
		frame.object_kp.filter(keypoint_filter_vec);
		cout << (boost::format("Filtered from %d to %d keypoints.") % num_keypoints % frame.object_kp.keypoints.size()).str() << endl;
	}

	// finally add projections of all keypoints
	frame.object_kp_projection_cloud_ptr = computeObjectKPProjectionCloud(frame);


	// debug features
	if (params.features_debug_images) {
		cv::Mat feature_image;
		cv::drawKeypoints( image_for_features, frame.object_kp.keypoints, feature_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		showInWindow("features", feature_image);
	}
}

bool ObjectModeler::alignWithFeaturesToModel(Eigen::Affine3f const& initial_pose, FrameT const& frame, KeypointsT const& model_keypoints, Eigen::Affine3f& result_pose, std::vector<cv::DMatch>& inlier_matches)
{
	inlier_matches.clear();
	bool success = false;

	if (model_keypoints.keypoints.empty()) {
		cout << "model_keypoints.keypoints.empty()" << endl;
		return false;
	}

	// align to model
	bool use_ratio_test = false;
	float ratio_test_ratio = 0.8;
	boost::scoped_ptr<cv::DescriptorMatcher> matcher_ptr;
	if (feature_type == FEATURE_TYPE_FAST) {
		matcher_ptr.reset(new cv::BFMatcher(cv::NORM_L1));
	}
	else if (feature_type == FEATURE_TYPE_SURF) {
		matcher_ptr.reset(new cv::BFMatcher(cv::NORM_L2));
	}
	else if (feature_type == FEATURE_TYPE_ORB) {
		matcher_ptr.reset(new cv::BFMatcher(cv::NORM_L1));
	}
	else {
		throw std::runtime_error("unknown feature type");
	}

	std::vector< cv::DMatch > matches;
	if (use_ratio_test) {
		std::vector<std::vector<cv::DMatch> > knn_matches;
		const int k = 2;
		matcher_ptr->knnMatch(frame.object_kp.descriptors, model_keypoints.descriptors, knn_matches, k);
		for (size_t i = 0; i < knn_matches.size(); i++) {
			if (knn_matches[i][0].distance / knn_matches[i][1].distance < ratio_test_ratio) {
				matches.push_back(knn_matches[i][0]);
			}
		}
	}
	else {
		matcher_ptr->match(frame.object_kp.descriptors, model_keypoints.descriptors, matches);
	}
	cout << "feature matches against model_keypoints: " << matches.size() << endl;

	// evidently PCL 1.6 won't let you extract duplicate points from a cloud...so I'll do it myself
	KeypointCloudT::Ptr frame_xyz_cloud(new KeypointCloudT);
	KeypointCloudT::Ptr model_xyz_cloud_untransformed(new KeypointCloudT);
	for (std::vector<cv::DMatch>::iterator match_iter = matches.begin(); match_iter != matches.end(); ++match_iter) {
		frame_xyz_cloud->push_back(frame.object_kp.keypoint_cloud->at(match_iter->queryIdx));
		model_xyz_cloud_untransformed->push_back(model_keypoints.keypoint_cloud->at(match_iter->trainIdx));
	}
	// perhaps unnecessary PCL bullshit:
	frame_xyz_cloud->header = frame.object_kp.keypoint_cloud->header;
	frame_xyz_cloud->width = frame_xyz_cloud->size();
	frame_xyz_cloud->height = 1;
	frame_xyz_cloud->is_dense = true;
	model_xyz_cloud_untransformed->header = model_keypoints.keypoint_cloud->header;
	model_xyz_cloud_untransformed->width = model_xyz_cloud_untransformed->size();
	model_xyz_cloud_untransformed->height = 1;
	model_xyz_cloud_untransformed->is_dense = true;

	// transform the model points by the current pose (so we are computing a relative pose)
	KeypointCloudT::Ptr model_xyz_cloud_transformed(new KeypointCloudT);
	pcl::transformPointCloud(*model_xyz_cloud_untransformed, *model_xyz_cloud_transformed, initial_pose);

	// TODO: use existing kp projection cloud
	KeypointCloudT::Ptr frame_projection_cloud(new KeypointCloudT);
	KeypointCloudT::Ptr model_projection_cloud(new KeypointCloudT); // stays empty...needed because of stupid existing interface
	frame_projection_cloud->resize(matches.size());
	for (unsigned int m = 0; m < matches.size(); m++) {
		const cv::DMatch & match = matches[m];
		const int & source_index = match.queryIdx;
		const cv::KeyPoint & source_cv_keypoint = frame.object_kp.keypoints[source_index];
		frame_projection_cloud->points[m].x = source_cv_keypoint.pt.x;
		frame_projection_cloud->points[m].y = source_cv_keypoint.pt.y;
	}
	g2o_stereo_projector.fillInZ(*frame_xyz_cloud, *frame_projection_cloud);

	///////////////////////////
	// model for ransac
	std::vector<int> inliers;
	Eigen::VectorXf coefs_before_optimization;
	Eigen::VectorXf coefs_optimized;
	bool fix_point_positions_in_sba = true;
	//setVerbosityLevel(pcl::console::L_DEBUG);

	// note that the target is the frame, and the source is the model!!
	pcl_peter::SampleConsensusModelRegistrationReprojection<KeypointPointT, KeypointPointT>::Ptr model(
		new pcl_peter::SampleConsensusModelRegistrationReprojection<KeypointPointT, KeypointPointT>(
		model_xyz_cloud_transformed,
		model_projection_cloud,
		g2o_stereo_projector,
		fix_point_positions_in_sba));
	model->setInputTarget(frame_xyz_cloud, frame_projection_cloud);

	//////////////////////////////
	// ransac
	pcl::StopWatch sw_ransac;
	pcl::RandomSampleConsensus<KeypointPointT>::Ptr ransac(new pcl::RandomSampleConsensus<KeypointPointT>(model));
	ransac->setDistanceThreshold(params.ransac_pixel_distance);
	ransac->setProbability(params.ransac_probability);
	ransac->setMaxIterations(params.ransac_max_iterations);
	pcl::StopWatch sw_compute_model;
	bool ransac_result = ransac->computeModel(params.ransac_verbosity);
	//cout << "[TIMING] ransac->computeModel() took " << sw_compute_model.getTime() << " ms." << endl;
	ransac->getInliers(inliers);
	cout << "Ransac inlier count: " << inliers.size() << endl;

	// always put the inliers in inlier matches (even if it fails the following min_inliers test)
	for (unsigned int i = 0; i < inliers.size(); i++) {
		inlier_matches.push_back(matches[inliers[i]]);
	}

	if (params.features_debug_images) {
		// also grab the keypoints that were inliers for visualization
		std::vector<cv::KeyPoint> inlier_keypoints;
		for (unsigned int i = 0; i < inlier_matches.size(); i++) {
			inlier_keypoints.push_back(frame.object_kp.keypoints[inlier_matches[i].queryIdx]);
		}
		cv::Mat inlier_keypoints_image;
		cv::drawKeypoints(frame.image_color, inlier_keypoints, inlier_keypoints_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		showInWindow("inlier_keypoints", inlier_keypoints_image);
	}

	if (ransac_result && inliers.size() >= params.ransac_min_inliers) {
		//////////////
		// optimize model
		ransac->getModelCoefficients(coefs_before_optimization);
		model->optimizeModelCoefficients(inliers, coefs_before_optimization, coefs_optimized);

		assert(coefs_optimized.size() == 16);
		Eigen::Affine3f transform;
		transform.matrix().row (0) = coefs_optimized.segment<4>(0);
		transform.matrix().row (1) = coefs_optimized.segment<4>(4);
		transform.matrix().row (2) = coefs_optimized.segment<4>(8);
		transform.matrix().row (3) = coefs_optimized.segment<4>(12);

		result_pose = transform * initial_pose;
		success = true;
	}
	else {
		result_pose = initial_pose;
		success = false;
		cout << "Ransac FAILED" << endl;
	}
	//cout << "[TIMING] Feature RANSAC took " << sw_ransac.getTime() << " ms." << endl;

	return success;
}

void ObjectModeler::updateModelKeypoints(const Eigen::Affine3f& object_pose, const FrameT& frame, const std::vector<cv::DMatch>& inlier_matches)
{
	Eigen::Affine3f inverse_object_pose = object_pose.inverse();

	if (params.features_frame_to_frame) {
		std::vector<bool> keep_frame_keypoint(frame.object_kp.keypoints.size(), true);

		// transform frame keypoints appropriately and replace all previous model kp
		Keypoints<KeypointPointT> new_model_kp = frame.object_kp;
		new_model_kp.filter(keep_frame_keypoint);
		pcl::transformPointCloud(*new_model_kp.keypoint_cloud, *new_model_kp.keypoint_cloud, inverse_object_pose);
		model_kp = new_model_kp;
	}
	else {
		if (params.use_patch_volumes) {
			cout << "you must use features_frame_to_frame when use_patch_volumes!!" << endl;
			exit(1);
		}

		// see which keypoints to keep
		// also update model_kp inlier count
		std::vector<bool> keep_frame_keypoint(frame.object_kp.keypoints.size(), true);
		for (size_t i = 0; i < inlier_matches.size(); i++) {
			// matches refer to frame and object kp (query and train, respectively)
			keep_frame_keypoint[inlier_matches[i].queryIdx] = false;
			model_kp.inlier_count[inlier_matches[i].trainIdx]++;
		}

		// also only keep frame keypoint if it falls in the volume
		// note this means we transform frame keypoints twice (see later cloud transform)
		for (size_t i = 0; i < keep_frame_keypoint.size(); i++) {
			Eigen::Vector3f p = inverse_object_pose * frame.object_kp.keypoint_cloud->at(i).getVector3fMap();
			bool is_in_volume = opencl_tsdf_ptr->isPointInVolume(p);
			if (!is_in_volume) {
				keep_frame_keypoint[i] = false;
			}
		}

		// Now remove all model_kp that have never been an inlier
		std::vector<bool> keep_model_keypoint(model_kp.keypoints.size(), false);
		for (size_t i = 0; i < model_kp.keypoints.size(); i++) {
			if (model_kp.inlier_count[i] > 0) {
				keep_model_keypoint[i] = true;
			}
		}

		// delete keypoints that have never been an inlier
		// note that the fresh points aren't in here yet
		model_kp.filter(keep_model_keypoint);

		// add in new
		Keypoints<KeypointPointT> new_model_kp = frame.object_kp;
		new_model_kp.filter(keep_frame_keypoint);
		pcl::transformPointCloud(*new_model_kp.keypoint_cloud, *new_model_kp.keypoint_cloud, inverse_object_pose);
		model_kp.append(new_model_kp);
	}

	///////////////////////////
	// Model keypoints for render
	CloudT::Ptr model_kp_to_view(new CloudT);
	pcl::copyPointCloud(*model_kp.keypoint_cloud, *model_kp_to_view);
	pcl::transformPointCloud(*model_kp_to_view, *model_kp_to_view, object_pose);
	for (size_t i = 0; i < model_kp_to_view->size(); i++) {
		PointT & p = model_kp_to_view->points[i];
		if (model_kp.inlier_count[i] > 0) {
			p.r = 0;
			p.b = 0;
			p.g = 255;
		}
		else {
			p.r = 0;
			p.b = 255;
			p.g = 0;
		}
	}
	tc_model_kp.setCloud(model_kp_to_view);
}

CloudT::Ptr ObjectModeler::getVolumePointsFromOpenCLTSDFForPose(OpenCLTSDF & tsdf, const Eigen::Affine3f& object_pose, bool show_max_points) const
{
	CloudT::Ptr nonzero_cloud(new CloudT);

	std::vector<std::pair<Eigen::Vector3f, float> > nonzero;
	float d_epsilon = 1e-6;
	if (show_max_points) d_epsilon = -1;
	tsdf.getNonzeroFilteredVoxelCenters(object_pose, d_epsilon, nonzero);
	for (size_t i = 0; i < nonzero.size(); i++) {
		PointT p;
		p.getVector3fMap() = nonzero[i].first;
		if (nonzero[i].second < 0) {
			p.r = 255;
			p.g = 0;
			p.b = 0;
		}
		else {
			p.r = 0;
			p.g = 255;
			p.b = 0;
		}
		nonzero_cloud->points.push_back(p);
	}

	return nonzero_cloud;
}

CloudT::Ptr ObjectModeler::getRidOfNaNs(CloudT::ConstPtr cloud)
{
	CloudT::Ptr result(new CloudT);
	for (size_t i = 0; i < cloud->size(); i++) {
		const PointT& p = cloud->points[i];
		if (pcl_isfinite(p.z)) {
			result->points.push_back(p);
		}
	}
	result->width = result->size();
	result->height = 1;
	return result;
}

KeypointCloudT::Ptr ObjectModeler::computeObjectKPProjectionCloud(const FrameT& frame)
{
	KeypointCloudT::Ptr projection_cloud(new KeypointCloudT);
	projection_cloud->resize(frame.object_kp.size());
	for (size_t i = 0; i < frame.object_kp.size(); i++) {
		const cv::KeyPoint & cv_keypoint = frame.object_kp.keypoints[i];
		KeypointPointT& p = projection_cloud->points[i];
		p.x = cv_keypoint.pt.x;
		p.y = cv_keypoint.pt.y;
	}
	g2o_stereo_projector.fillInZ(*frame.object_kp.keypoint_cloud, *projection_cloud);
	return projection_cloud;
}

void ObjectModeler::renderVolumeWithPose(const Eigen::Affine3f& custom_object_pose, const Eigen::Vector3f& light_direction, float scale, cv::Mat& result_colors, cv::Mat& result_normals, cv::Mat& result_depth)
{
	Eigen::Vector2f render_f(params.camera_focal_x * scale, params.camera_focal_y * scale);
	Eigen::Vector2f render_c(params.camera_center_x * scale, params.camera_center_y * scale);
	cv::Rect render_rect(0, 0, params.camera_size_x * scale, params.camera_size_y * scale);
	float render_min_depth = params.render_min_depth;
	float render_max_depth = params.render_max_depth;


	RenderBuffers render_buffers (*cl_ptr);
	renderVolumeWithOpenCL(custom_object_pose, render_f, render_c, render_min_depth, render_max_depth, render_rect, render_buffers);
	
	// get clouds
	CloudT::Ptr render_cloud(new CloudT);
	pcl::PointCloud<pcl::Normal>::Ptr render_normal_cloud(new pcl::PointCloud<pcl::Normal>);
	extractRenderBuffersToClouds(render_rect, render_buffers, *render_cloud, *render_normal_cloud);

	result_colors = cloudToImage(*render_cloud, cv::Vec3b(0,0,255));
	result_normals = normalCloudToImage(*render_normal_cloud, light_direction, false, cv::Vec3b(0,0,255));
	result_depth = cloudToColorDepthImage(*render_cloud, cv::Vec3b(0,0,255));
}


void ObjectModeler::renderVolumeWithOpenCL(const Eigen::Affine3f& object_pose, const Eigen::Vector2f& proj_f, const Eigen::Vector2f& proj_c, float render_min_depth, float render_max_depth, const cv::Rect& render_rect, 
	RenderBuffers & render_buffers)
{
	render_buffers.setSize(render_rect.width, render_rect.height);
	render_buffers.resetAllBuffers();

	opencl_tsdf_ptr->renderFrame(object_pose, proj_f.x(), proj_f.y(), proj_c.x(), proj_c.y(), render_min_depth, render_max_depth,
			render_rect.x, render_rect.y, render_rect.width, render_rect.height, false, 1,
			render_buffers);
}

void ObjectModeler::extractRenderBuffersToClouds(cv::Rect const& render_rect, RenderBuffers const& render_buffers,
	CloudT& point_cloud, pcl::PointCloud<pcl::Normal>& normal_cloud)
{
	int rows = render_rect.height;
	int cols = render_rect.width;
	int image_size = rows * cols;
	std::vector<int> render_mask(image_size);
	std::vector<float> render_points(image_size * 4);
	std::vector<float> render_normals(image_size * 4);
	std::vector<unsigned char> render_colors(image_size * 4); // assumes the bgra that we always use
	render_buffers.readAllBuffers(render_mask.data(), render_points.data(), render_normals.data(), render_colors.data());

	// fill in cloud and normal_cloud as expected
	point_cloud.points.resize(rows * cols);
	point_cloud.width = cols;
	point_cloud.height = rows;
	point_cloud.points.resize(point_cloud.width * point_cloud.height);
	point_cloud.is_dense = false;

	normal_cloud.points.resize(rows * cols);
	normal_cloud.width = cols;
	normal_cloud.height = rows;
	normal_cloud.points.resize(normal_cloud.width * normal_cloud.height);
	normal_cloud.is_dense = false;

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			int image_index = row * cols + col;
			PointT& p = point_cloud.at(col, row);
			pcl::Normal& n = normal_cloud.at(col, row);
			if (render_mask[image_index]) {
				int image_index_4 = image_index * 4;
				p.x = render_points[image_index_4];
				p.y = render_points[image_index_4+1];
				p.z = render_points[image_index_4+2];
				p.b = render_colors[image_index_4];
				p.g = render_colors[image_index_4+1];
				p.r = render_colors[image_index_4+2];
				n.normal_x = render_normals[image_index_4];
				n.normal_y = render_normals[image_index_4+1];
				n.normal_z = render_normals[image_index_4+2];
			}
			else {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
				n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}
}

void ObjectModeler::renderPatchVolumesWithPose(const Eigen::Affine3f& custom_object_pose, const Eigen::Vector3f& light_direction, float scale,
	cv::Mat& result_colors, cv::Mat& result_normals, cv::Mat& result_depth, cv::Mat& result_segments)
{
	Eigen::Vector2f render_f(params.camera_focal_x * scale, params.camera_focal_y * scale);
	Eigen::Vector2f render_c(params.camera_center_x * scale, params.camera_center_y * scale);
	cv::Rect render_rect(0, 0, params.camera_size_x * scale, params.camera_size_y * scale);

	RenderBuffers render_buffers(*cl_ptr);
	renderPatchVolumes(custom_object_pose, render_f, render_c, params.render_min_depth, params.render_max_depth, render_rect, -1 /* min_age */, -1 /* max_age */, -1 /*max_normal_angle*/, true, false, render_buffers);

	// get clouds
	CloudT::Ptr render_image_cloud_opencl(new CloudT);
	pcl::PointCloud<pcl::Normal>::Ptr render_image_normal_cloud_opencl(new pcl::PointCloud<pcl::Normal>);
	extractRenderBuffersToClouds(render_rect, render_buffers, *render_image_cloud_opencl, *render_image_normal_cloud_opencl);
	result_segments = cv::Mat (render_rect.size(), CV_32S);
	render_buffers.readRenderMask((int*)result_segments.data);

	// make images
	const static cv::Vec3b nan_color(0,0,255);
	result_colors = cloudToImage(*render_image_cloud_opencl, nan_color);
	result_normals = normalCloudToImage(*render_image_normal_cloud_opencl, light_direction, false, nan_color);
	result_depth = cloudToColorDepthImage(*render_image_cloud_opencl, nan_color);
}

void ObjectModeler::renderPatchVolumes(const Eigen::Affine3f& object_pose, const Eigen::Vector2f& proj_f, const Eigen::Vector2f& proj_c, float render_min_depth, float render_max_depth, const cv::Rect& render_rect,
	int min_age, int max_age, float max_normal_angle, bool deallocate_after, bool update_frame_in_frustum, 
	RenderBuffers & render_buffers)
{
	// go through and render each patch volume, replacing values in output if closer to camera
	const float min_dot_product = acos(M_PI / 180.0 * max_normal_angle);
	const static Eigen::Vector3f normal_at_camera(0,0,-1);

	render_buffers.setSize(render_rect.width, render_rect.height);
	render_buffers.resetAllBuffers();

	const Eigen::Vector2f resolution(params.camera_size_x, params.camera_size_y);
	//const Eigen::Vector2f proj_f(params.camera_focal_x, params.camera_focal_y);
	//const Eigen::Vector2f proj_c(params.camera_center_x, params.camera_center_y);
	// TODO: use this
	Frustum frustum_in_world(object_pose.inverse(), resolution, proj_f, proj_c, render_min_depth, render_max_depth);

	int pvs_actually_rendered = 0;
	for (int c = 1; c < pv_list.size(); ++c) {
		int age_to_use = input_frame_counter - pv_list.at(c)->frame_last_in_frustum;
		if (min_age >= 0 && age_to_use < min_age) continue;
		if (max_age >= 0 && age_to_use > max_age) continue;
		if (max_normal_angle > 0) {
			Eigen::Vector3f pv_normal_camera_frame = (object_pose * pv_list.at(c)->pose).linear() * pv_list.at(c)->normal;
			float this_dot_product = normal_at_camera.dot(pv_normal_camera_frame);

			if (params.pv_verbose) {
				cout << "pv_normal_camera_frame: " << pv_normal_camera_frame.transpose() << endl;
				cout << "this_dot_product: " << this_dot_product << endl;
				cout << "min_dot_product: " << min_dot_product << endl;
			}

			if (this_dot_product < min_dot_product) continue;
		}

		// frustum check
		Eigen::Array3f bb_min, bb_max;
		pv_list.at(c)->tsdf_ptr->getAABB(bb_min, bb_max);
		Frustum frustum((object_pose * pv_list.at(c)->pose).inverse(), resolution, proj_f, proj_c, render_min_depth, render_max_depth);
		if (!frustum.doesAABBIntersect(bb_min.matrix(), bb_max.matrix())) {
			if (params.pv_verbose) {
				cout << "Skipping out-of-frustum PV: " << c << endl;
			}
			continue;
		}

		if (update_frame_in_frustum) {
			pv_list.at(c)->frame_last_in_frustum = input_frame_counter;
		}

		// now reallocate if not allocated
		// don't NEED this anymore, but allows us to manage the loop vs sequential vs user a little bit better
		bool deallocate_this_after = false;
		if (!pv_list.at(c)->tsdf_ptr->buffersAreAllocated()) {
			pv_list.at(c)->tsdf_ptr->reallocateVolumeBuffers();

			deallocate_this_after = deallocate_after;

			if (params.pv_verbose) {
				cout << "reallocateVolumeBuffers() " << c << endl;
			}

			if (params.pv_debug_update_visualizer) {
				updatePatchVolumeVisualizer();
			}
		}

		if (params.pv_verbose) {
			cout << "rendering patch volume: " << c << endl;
		}

		pv_list.at(c)->tsdf_ptr->renderFrame(object_pose * pv_list.at(c)->pose, proj_f.x(), proj_f.y(), proj_c.x(), proj_c.y(), render_min_depth, render_max_depth,
			render_rect.x, render_rect.y, render_rect.width, render_rect.height, true, c,
			render_buffers);

		pvs_actually_rendered++;

		if (deallocate_this_after) {
			pv_list.at(c)->tsdf_ptr->deallocateVolumeBuffers();
		}

		deallocateUntilUnderMaxSize();
	} // loop over pvs

	// done!
	rs_pvs_rendered.push(pvs_actually_rendered);
	cout << "pvs_actually_rendered: " << rs_pvs_rendered.summary() << endl;
}

void ObjectModeler::deallocateUntilUnderMaxSize()
{
	if (params.pv_max_mb_allocated > 0) {
		for (int c = 1; c < pv_list.size(); ++c) {
			if (getAllocatedPatchVolumeSize() < params.pv_max_mb_allocated) break;
			pv_list.at(c)->tsdf_ptr->deallocateVolumeBuffers();
		}
	}
}

void ObjectModeler::writeTablesOfValues(fs::path folder)
{
	for (TablesOfValuesT::iterator map_iter = tables_of_values.begin(); map_iter != tables_of_values.end(); ++map_iter) {
		fs::path filename = folder / ("tov-" + map_iter->first + ".txt");
		std::ofstream ofs (filename.string().c_str());
		std::ostream_iterator<float> output_iterator(ofs, "\n");
		std::copy(map_iter->second.begin(), map_iter->second.end(), output_iterator);			
	}
}

float ObjectModeler::getLastDelta(std::string name)
{
	if (tables_of_values.find(name) == tables_of_values.end()) throw std::runtime_error ( (boost::format("TOV name fail: %s") % name).str().c_str() );
	float result = 0;
	size_t table_size = tables_of_values[name].size();
	if (table_size >= 2) {
		float v1 = tables_of_values[name][table_size - 2];
		float v2 = tables_of_values[name][table_size - 1];
		result = v2 - v1;
	}
	return result;
}

float ObjectModeler::getRecentMean(std::string name, size_t count)
{
	if (tables_of_values.find(name) == tables_of_values.end()) throw std::runtime_error ( (boost::format("TOV name fail: %s") % name).str().c_str() );
	float result_sum = 0;
	size_t result_count = 0;
	size_t table_size = tables_of_values[name].size();
	for (size_t i = table_size - 1; i >= 0 && (table_size - i <= count); --i) {
		result_sum += tables_of_values[name][i];
		result_count++;
	}
	return (result_sum / result_count);
}

// deprecated (in tsdf)
void ObjectModeler::generateMeshForTSDF(OpenCLTSDF & tsdf, VertexCloudT::Ptr & result_vertex_cloud, TrianglesPtrT & result_triangles)
{
	cout << "Obtaining Buffers..." << endl;
	std::vector<float> bufferD;
	std::vector<float> bufferDW;
	std::vector<unsigned char> bufferC;
	std::vector<float> bufferCW;
	tsdf.getAllBuffers(bufferD, bufferDW, bufferC, bufferCW);

	cout << "Running marching cubes..." << endl;
	CIsoSurface<float> iso_surface;
	float cell_size = tsdf.getVolumeCellSize();
	if (params.mesh_marching_cubes_weights) {
		iso_surface.GenerateSurfaceWithWeights(bufferD.data(), bufferDW.data(), 0, tsdf.getCellCountX() - 1, tsdf.getCellCountY() - 1, tsdf.getCellCountZ() - 1, cell_size, cell_size, cell_size);
	}
	else {
		iso_surface.GenerateSurface(bufferD.data(), 0, tsdf.getCellCountX() - 1, tsdf.getCellCountY() - 1, tsdf.getCellCountZ() - 1, cell_size, cell_size, cell_size);
	}

	cout << "Converting to PCL mesh and assigning colors..." << endl;
	typedef pcl::PointCloud<pcl::PointXYZRGBNormal> VertexCloudT;
	VertexCloudT::Ptr vertex_cloud_ptr (new VertexCloudT);
	vertex_cloud_ptr->resize(iso_surface.m_nVertices);
	for (unsigned int i = 0; i < vertex_cloud_ptr->size(); i++) {
		pcl::PointXYZRGBNormal &p = vertex_cloud_ptr->at(i);
		p.getVector3fMap() = Eigen::Vector3f::Map(iso_surface.m_ppt3dVertices[i]);
		p.getNormalVector3fMap() = Eigen::Vector3f::Map(iso_surface.m_pvec3dNormals[i]);
		Eigen::Vector3f voxel_float;
		voxel_float[0] = p.x / cell_size;
		voxel_float[1] = p.y / cell_size;
		voxel_float[2] = p.z / cell_size;
		Eigen::Matrix<unsigned char, 4, 1> color_cv = interpolateColorForMesh(tsdf, bufferC, voxel_float);
		p.b = color_cv[0];
		p.g = color_cv[1];
		p.r = color_cv[2];
	}

	boost::shared_ptr<std::vector<pcl::Vertices> >  triangles_ptr (new std::vector<pcl::Vertices>);
	triangles_ptr->resize(iso_surface.m_nTriangles);
	for (unsigned int i = 0; i < triangles_ptr->size(); i++) {
		triangles_ptr->at(i).vertices.push_back(iso_surface.m_piTriangleIndices[3*i]);
		triangles_ptr->at(i).vertices.push_back(iso_surface.m_piTriangleIndices[3*i+1]);
		triangles_ptr->at(i).vertices.push_back(iso_surface.m_piTriangleIndices[3*i+2]);
	}

	if (params.mesh_marching_cubes_weights) {
		// NOTHING!
	}
	else {
		// remove vertices that have any neighboring cell with 0 weight
		std::set<int> bogus_vertices;
		for (uint32_t i = 0; i < vertex_cloud_ptr->size(); i++) {
			pcl::PointXYZRGBNormal &p = vertex_cloud_ptr->at(i);
			Eigen::Vector3f voxel_float = p.getVector3fMap() / cell_size;
			// this (used to?) produce "black" edge vertices:
			bool is_bogus_vertex = !allSurroundingVerticesNonzeroOrOutsideVolume(tsdf, bufferDW, voxel_float);
			//bool is_bogus_vertex = !allSurroundingVerticesNonzero(tsdf, bufferW, voxel_float);
			if (is_bogus_vertex) {
				bogus_vertices.insert(i);
			}
		}

		// now have to remove vertices (and triangles which point to them)
		VertexCloudT::Ptr new_vertex_cloud_ptr (new VertexCloudT);
		std::map<uint32_t, uint32_t> vertex_map;
		for (uint32_t i = 0; i < vertex_cloud_ptr->size(); i++) {
			// if not bogus, copy
			if (bogus_vertices.find(i) == bogus_vertices.end()) {
				pcl::PointXYZRGBNormal &p = vertex_cloud_ptr->at(i);
				new_vertex_cloud_ptr->push_back(p);
				vertex_map[i] = new_vertex_cloud_ptr->size() - 1;
			}
		}

		// and remove / remap triangles
		boost::shared_ptr<std::vector<pcl::Vertices> >  new_triangles_ptr (new std::vector<pcl::Vertices>);
		for (uint32_t i = 0; i < triangles_ptr->size(); i++) {
			pcl::Vertices &old_triangle = triangles_ptr->at(i);
			if (bogus_vertices.find(old_triangle.vertices[0]) == bogus_vertices.end() && 
				bogus_vertices.find(old_triangle.vertices[1]) == bogus_vertices.end() &&
				bogus_vertices.find(old_triangle.vertices[2]) == bogus_vertices.end()) {
					pcl::Vertices new_triangle;
					new_triangle.vertices.resize(3);
					new_triangle.vertices[0] = vertex_map[old_triangle.vertices[0]];
					new_triangle.vertices[1] = vertex_map[old_triangle.vertices[1]];
					new_triangle.vertices[2] = vertex_map[old_triangle.vertices[2]];
					new_triangles_ptr->push_back(new_triangle);
			}
		}
		vertex_cloud_ptr = new_vertex_cloud_ptr;
		triangles_ptr = new_triangles_ptr;
	}

	// moves vertices by offset to center
	Eigen::Vector3f center_offset = (tsdf.getVolumeOrigin().cast<float>()) * cell_size;

	for (unsigned int i = 0; i < vertex_cloud_ptr->size(); i++) {
		pcl::PointXYZRGBNormal &p = vertex_cloud_ptr->at(i);
		p.getVector3fMap() += center_offset;
	}

	result_vertex_cloud = vertex_cloud_ptr;
	result_triangles = triangles_ptr;
}

// Ugly..first two are both input and output params...violates google style ;)
void ObjectModeler::appendToMesh(VertexCloudT::Ptr & main_vertices_ptr, TrianglesPtrT & main_triangles_ptr, VertexCloudT::Ptr const& vertices_to_add_ptr, TrianglesPtrT const& triangles_to_add_ptr)
{
	size_t vertex_offset = main_vertices_ptr->size();

	main_vertices_ptr->insert(main_vertices_ptr->end(), vertices_to_add_ptr->begin(), vertices_to_add_ptr->end());
	// push modified triangles individually
	for (size_t i = 0; i < triangles_to_add_ptr->size(); ++i) {
		pcl::Vertices const& old_triangle = triangles_to_add_ptr->at(i);
		main_triangles_ptr->push_back(pcl::Vertices());
		main_triangles_ptr->back().vertices.push_back(old_triangle.vertices[0] + vertex_offset);
		main_triangles_ptr->back().vertices.push_back(old_triangle.vertices[1] + vertex_offset);
		main_triangles_ptr->back().vertices.push_back(old_triangle.vertices[2] + vertex_offset);
	}
}

void ObjectModeler::saveMesh(fs::path filename, VertexCloudT::Ptr const& vertex_cloud_ptr, TrianglesPtrT const& triangles_ptr)
{
	pcl::PolygonMesh polygon_mesh;
	pcl::toROSMsg(*vertex_cloud_ptr, polygon_mesh.cloud);
	polygon_mesh.polygons = *triangles_ptr;
	if (!pcl::io::savePLYFile(filename.string(), polygon_mesh, 5)) cout << "Saved mesh to: " << filename << endl;
	else cout << "FAILED to save mesh to: " << filename << endl;
}

void ObjectModeler::generateMesh()
{
	boost::mutex::scoped_lock lock(mutex_volumes);
	boost::mutex::scoped_lock lock2(mutex_pv_meshes_show);

	tm_pv_show_generated_list.clear();
	tc_pv_compare_to_mesh_list.clear();

	if (params.use_patch_volumes) {
		fs::path dump_path_no_loop = prepareDumpPath(fs::path("pv_meshes_no_loop") / (boost::format("%05d") % output_frame_counter).str());
		fs::path dump_path_loop = prepareDumpPath(fs::path("pv_meshes_loop") / (boost::format("%05d") % output_frame_counter).str());

		// Gotta combine into single mesh updating vertex pointers...
		VertexCloudT::Ptr vertex_cloud_combined_no_loop_ptr (new VertexCloudT);
		TrianglesPtrT triangles_combined_no_loop_ptr (new TrianglesT);
		VertexCloudT::Ptr vertex_cloud_combined_loop_ptr (new VertexCloudT);
		TrianglesPtrT triangles_combined_loop_ptr (new TrianglesT);

		for (int c = 1; c < pv_list.size(); ++c) {
			VertexCloudT::Ptr vertex_cloud_ptr;
			TrianglesPtrT triangles_ptr;
			
			generateMeshForTSDF(*(pv_list.at(c)->tsdf_ptr), vertex_cloud_ptr, triangles_ptr);

			if (!vertex_cloud_ptr) {
				cout << "Skipping NULL vertex cloud ptr" << endl;
				continue;
			}

			if (triangles_ptr->empty()) {
				cout << "Skipping empty volume " << c << endl;
				continue;
			}

			VertexCloudT::Ptr transformed_vertex_cloud_ptr(new VertexCloudT);
			pcl::transformPointCloud(*vertex_cloud_ptr, *transformed_vertex_cloud_ptr, pv_list.at(c)->original_pose);
			fs::path dump_filename = dump_path_no_loop / ((boost::format("mesh-%05d-pv-%04d.ply") % output_frame_counter % c).str());
			saveMesh(dump_filename, transformed_vertex_cloud_ptr, triangles_ptr);

			appendToMesh(vertex_cloud_combined_no_loop_ptr, triangles_combined_no_loop_ptr, transformed_vertex_cloud_ptr, triangles_ptr);

			if (params.pv_loop_closure) {
				pcl::transformPointCloud(*vertex_cloud_ptr, *transformed_vertex_cloud_ptr, pv_list.at(c)->pose);
				fs::path dump_filename = dump_path_loop / ((boost::format("mesh-%05d-pv-%04d.ply") % output_frame_counter % c).str());
				saveMesh(dump_filename, transformed_vertex_cloud_ptr, triangles_ptr);

				appendToMesh(vertex_cloud_combined_loop_ptr, triangles_combined_loop_ptr, transformed_vertex_cloud_ptr, triangles_ptr);
			}

			if (params.mesh_show) {
				// apply object pose, dingus
				VertexCloudT::Ptr v(new VertexCloudT);
				pcl::transformPointCloud(*vertex_cloud_ptr, *v, object_pose * pv_list.at(c)->pose);
				std::string mesh_name = (boost::format("pv_mesh_%05d")%c).str();
				tm_pv_show_generated_list.push_back(boost::shared_ptr<ToggleMesh<pcl::PointXYZRGBNormal> >());
				tm_pv_show_generated_list.back().reset(new ToggleMesh<pcl::PointXYZRGBNormal>(mesh_name, true));
				tm_pv_show_generated_list.back()->setCloud(v, triangles_ptr, 1.0);

				// show the voxels as well
				// yeah, this is slow:
				std::string cloud_name = (boost::format("pv_cloud_%05d")%c).str();
				tc_pv_compare_to_mesh_list.push_back(boost::shared_ptr<ToggleCloud<PointT> >());
				tc_pv_compare_to_mesh_list.back().reset(new ToggleCloud<PointT>(cloud_name, true));
				CloudT::Ptr voxel_points = getVolumePointsFromOpenCLTSDFForPose(*pv_list.at(c)->tsdf_ptr, object_pose * pv_list.at(c)->pose, params.volume_debug_show_max_points);
				tc_pv_compare_to_mesh_list.back()->setCloud(voxel_points);
			}

			deallocateUntilUnderMaxSize();
		}

		// can now save fatty mesh too
		fs::path dump_filename = dump_path_no_loop / ((boost::format("mesh-%05d-all.ply") % output_frame_counter).str());
		saveMesh(dump_filename, vertex_cloud_combined_no_loop_ptr, triangles_combined_no_loop_ptr);
		if (params.pv_loop_closure) {
			dump_filename = dump_path_loop / ((boost::format("mesh-%05d-all.ply") % output_frame_counter).str());
			saveMesh(dump_filename, vertex_cloud_combined_loop_ptr, triangles_combined_loop_ptr);
		}
	}
	else {
		fs::path dump_path = prepareDumpPath("mesh");

		VertexCloudT::Ptr vertex_cloud_ptr;
		TrianglesPtrT triangles_ptr;

		generateMeshForTSDF(*opencl_tsdf_ptr, vertex_cloud_ptr, triangles_ptr);

		fs::path dump_filename = dump_path / ((boost::format("mesh-%05d.ply") % output_frame_counter).str());
		saveMesh(dump_filename, vertex_cloud_ptr, triangles_ptr);

		if (params.mesh_show) {
			// apply object pose, dingus
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr v(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::transformPointCloud(*vertex_cloud_ptr, *v, object_pose);
			std::string mesh_name = (boost::format("single_mesh")).str();
			tm_pv_show_generated_list.push_back(boost::shared_ptr<ToggleMesh<pcl::PointXYZRGBNormal> >());
			tm_pv_show_generated_list.back().reset(new ToggleMesh<pcl::PointXYZRGBNormal>(mesh_name, true));
			tm_pv_show_generated_list.back()->setCloud(v, triangles_ptr, 1.0);

			// yeah, this is slow:
			std::string cloud_name = (boost::format("single_mesh_cloud")).str();
			tc_pv_compare_to_mesh_list.push_back(boost::shared_ptr<ToggleCloud<PointT> >());
			tc_pv_compare_to_mesh_list.back().reset(new ToggleCloud<PointT>(cloud_name, true));
			CloudT::Ptr voxel_points = getVolumePointsFromOpenCLTSDFForPose(*opencl_tsdf_ptr, object_pose, params.volume_debug_show_max_points);
			tc_pv_compare_to_mesh_list.back()->setCloud(voxel_points);
		}
	}

	cout << "Done with generateMesh()" << endl;
}

bool ObjectModeler::allPossibleEdgeVerticesNonzero(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const
{
	// for meshing...for a point from marching cubes
	// if the floor(p+epsilon) and that +1 in all 3 coordinates has nonzero weight, leave the point
	// this is conservative...really should only check the two endpoints that generated it
	if (!tsdf.checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return false;
	const static float epsilon = 1e-6;
	const static Eigen::Vector3f epsilon_v(epsilon);
	Eigen::Vector3i floor_corner = tsdf.floorVector3fToInt( voxel_coords_f + epsilon_v );
	Eigen::Vector3f offset = voxel_coords_f - floor_corner.cast<float>(); // do i need this?

	const float w_000 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2])];
	const float w_001 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2]+1)];
	const float w_010 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2])];
	const float w_100 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2])];

	bool result = (
		(w_000 > 0) && 
		(w_001 > 0) && 
		(w_010 > 0) &&
		(w_100 > 0) );
	return result;
}

// deprecated (in tsdf)
bool ObjectModeler::allSurroundingVerticesNonzero(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const
{
	// for meshing...for a point from marching cubes
	if (!tsdf.checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return false;
	Eigen::Vector3i floor_corner = tsdf.floorVector3fToInt( voxel_coords_f );

	const float w_000 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2])];
	const float w_001 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2]+1)];
	const float w_010 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2])];
	const float w_011 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2]+1)];
	const float w_100 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2])];
	const float w_101 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2]+1)];
	const float w_110 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2])];
	const float w_111 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2]+1)];
									
	bool result = (
		(w_000 > 0) && 
		(w_001 > 0) && 
		(w_010 > 0) &&
		(w_011 > 0) &&
		(w_100 > 0) && 
		(w_101 > 0) && 
		(w_110 > 0) &&
		(w_111 > 0) );
	return result;
}

// deprecated (in tsdf)
bool ObjectModeler::allSurroundingVerticesNonzeroOrOutsideVolume(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const
{
	// for meshing...for a point from marching cubes

	// THIS IS WRONG...FIX IT!
	if (!tsdf.checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return true;

	Eigen::Vector3i floor_corner = tsdf.floorVector3fToInt( voxel_coords_f );

	const float w_000 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2])];
	const float w_001 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2]+1)];
	const float w_010 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2])];
	const float w_011 = weight_vector[tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2]+1)];
	const float w_100 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2])];
	const float w_101 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2]+1)];
	const float w_110 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2])];
	const float w_111 = weight_vector[tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2]+1)];
									
	bool result = (
		(w_000 > 0) && 
		(w_001 > 0) && 
		(w_010 > 0) &&
		(w_011 > 0) &&
		(w_100 > 0) && 
		(w_101 > 0) && 
		(w_110 > 0) &&
		(w_111 > 0) );
	return result;
}

// DEPRECATED
#if 0
Eigen::Matrix<unsigned char, 4, 1> ObjectModeler::interpolateColorForMesh(OpenCLTSDF const& tsdf, const std::vector<unsigned char> & color_vector, const Eigen::Vector3f& voxel_coords_f)
{
	typedef Eigen::Matrix<unsigned char, 4, 1> Vector4b;
	const static Eigen::Vector4f add_round(0.5,0.5,0.5,0.5);
	const static Eigen::Array4f min_array(0,0,0,0);
	const static Eigen::Array4f max_array(255,255,255,255);

	if (!tsdf.checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return Vector4b(0,0,0,0);

	Eigen::Vector3i floor_corner = tsdf.floorVector3fToInt(voxel_coords_f);
	Eigen::Vector3f offset = voxel_coords_f - floor_corner.cast<float>();

	const Eigen::Map<const Vector4b> d_000 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2])]);
	const Eigen::Map<const Vector4b> d_001 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0], floor_corner[1], floor_corner[2]+1)]);
	const Eigen::Map<const Vector4b> d_010 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2])]);
	const Eigen::Map<const Vector4b> d_011 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0], floor_corner[1]+1, floor_corner[2]+1)]);
	const Eigen::Map<const Vector4b> d_100 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2])]);
	const Eigen::Map<const Vector4b> d_101 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1], floor_corner[2]+1)]);
	const Eigen::Map<const Vector4b> d_110 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2])]);
	const Eigen::Map<const Vector4b> d_111 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2]+1)]);

	float off_x = offset[0];
	float off_y = offset[1];
	float off_z = offset[2];

	// this could over/underflow on final cast?
	Eigen::Vector4f result_float = (  d_000.cast<float>() * (1 - off_x) * (1 - off_y) * (1 - off_z)
									+ d_001.cast<float>() * (1 - off_x) * (1 - off_y) * (off_z)
									+ d_010.cast<float>() * (1 - off_x) * (off_y) * (1 - off_z)
									+ d_011.cast<float>() * (1 - off_x) * (off_y) * (off_z)
									+ d_100.cast<float>() * (off_x) * (1 - off_y) * (1 - off_z)
									+ d_101.cast<float>() * (off_x) * (1 - off_y) * (off_z)
									+ d_110.cast<float>() * (off_x) * (off_y) * (1 - off_z)
									+ d_111.cast<float>() * (off_x) * (off_y) * (off_z) );
	Eigen::Vector4f result_to_cast = result_float + add_round;
	result_to_cast = result_to_cast.array().max(min_array).matrix();
	result_to_cast = result_to_cast.array().min(max_array).matrix();
	Vector4b result = result_to_cast.cast<unsigned char>();

	return result;
}
#endif

// deprecated (in tsdf)
Eigen::Matrix<unsigned char, 4, 1> ObjectModeler::interpolateColorForMesh(OpenCLTSDF const& tsdf, const std::vector<unsigned char> & color_vector, const Eigen::Vector3f& voxel_coords_f)
{
	typedef Eigen::Matrix<unsigned char, 4, 1> Vector4b;
	const static Eigen::Vector4f add_round(0.5,0.5,0.5,0.5);
	const static Eigen::Array4f min_array(0,0,0,0);
	const static Eigen::Array4f max_array(255,255,255,255);

	//if (!tsdf.checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return Vector4b(0,0,0,0);

	Eigen::Vector3i fc = tsdf.floorVector3fToInt(voxel_coords_f);

	// set to 1 if inside volume
	// This will allow an adjustment for < 8 valid vertices
	float w_000 = 0;
	float w_001 = 0;
	float w_010 = 0;
	float w_011 = 0;
	float w_100 = 0;
	float w_101 = 0;
	float w_110 = 0;
	float w_111 = 0;

	Vector4b d_000;
	Vector4b d_001;
	Vector4b d_010;
	Vector4b d_011;
	Vector4b d_100;
	Vector4b d_101;
	Vector4b d_110;
	Vector4b d_111;

	if (tsdf.isVertexInVolume(fc[0], fc[1], fc[2])) {
		d_000 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0], fc[1], fc[2])]);
		w_000 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0], fc[1], fc[2]+1)) {
		d_001 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0], fc[1], fc[2]+1)]);
		w_001 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0], fc[1]+1, fc[2])) {
		d_010 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0], fc[1]+1, fc[2])]);
		w_010 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0], fc[1]+1, fc[2]+1)) {
		d_011 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0], fc[1]+1, fc[2]+1)]);
		w_011 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0]+1, fc[1], fc[2])) {
		d_100 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0]+1, fc[1], fc[2])]);
		w_100 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0]+1, fc[1], fc[2]+1)) {
		d_101 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0]+1, fc[1], fc[2]+1)]);
		w_101 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0]+1, fc[1]+1, fc[2])) {
		d_110 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0]+1, fc[1]+1, fc[2])]);
		w_110 = 1;
	}
	if (tsdf.isVertexInVolume(fc[0]+1, fc[1]+1, fc[2]+1)) {
		d_111 = Vector4b::Map(&color_vector[4 * tsdf.getBufferIndex(fc[0]+1, fc[1]+1, fc[2]+1)]);
		w_111 = 1;
	}

	Eigen::Vector3f offset = voxel_coords_f - fc.cast<float>();
	float off_x = offset[0];
	float off_y = offset[1];
	float off_z = offset[2];

	Eigen::Vector4f result_float(0,0,0,0);
	w_000 *= (1 - off_x) * (1 - off_y) * (1 - off_z);
	w_001 *= (1 - off_x) * (1 - off_y) * (off_z);
	w_010 *= (1 - off_x) * (off_y) * (1 - off_z);
	w_011 *= (1 - off_x) * (off_y) * (off_z);
	w_100 *= (off_x) * (1 - off_y) * (1 - off_z);
	w_101 *= (off_x) * (1 - off_y) * (off_z);
	w_110 *= (off_x) * (off_y) * (1 - off_z);
	w_111 *= (off_x) * (off_y) * (off_z);

	result_float += ( w_000 * d_000.cast<float>()
					+ w_001 * d_001.cast<float>()
					+ w_010 * d_010.cast<float>()
					+ w_011 * d_011.cast<float>()
					+ w_100 * d_100.cast<float>()
					+ w_101 * d_101.cast<float>()
					+ w_110 * d_110.cast<float>()
					+ w_111 * d_111.cast<float>() );
	// correct for missing weights
	float weight_sum = w_000
					 + w_001
					 + w_010
					 + w_011
					 + w_100
					 + w_101
					 + w_110
					 + w_111;
	if (weight_sum > 0) {
		result_float *= 1.f / weight_sum;
	}

	Eigen::Vector4f result_to_cast = result_float + add_round;
	result_to_cast = result_to_cast.array().max(min_array).matrix();
	result_to_cast = result_to_cast.array().min(max_array).matrix();
	Vector4b result = result_to_cast.cast<unsigned char>();

	return result;
}

void ObjectModeler::showVolumeEdges()
{
	if (params.use_patch_volumes) {
		cout << "showVolumeEdges not compatible with use_patch_volumes!" << endl;
		exit(1);
	}
	// get corner points
	std::vector<Eigen::Vector3f> corners = opencl_tsdf_ptr->getVolumeCorners(object_pose);
	// show the edges
	tls_volume_corners.setLineSet(getLineSetEdges(corners));
}

bool ObjectModeler::loadBothHistograms()
{
	bool result = true;
	if (!loadHistogram(params.mask_hand_hist_filename.string(), mask_object.histogram_hand)) {
		cout << "Failed to load hand histogram: " << params.mask_hand_hist_filename << endl;
		result = false;
	}
	if (!loadHistogram(params.mask_object_hist_filename.string(), mask_object.histogram_object)) {
		cout << "Failed to load object histogram: " << params.mask_object_hist_filename << endl;
		result = false;
	}
	return result;
}

cv::Mat ObjectModeler::randomlyColorSegments(const cv::Mat& segment_image, const cv::Mat& mask_image)
{
	prepareSegmentColorMap(segment_image);

	cv::Mat result(segment_image.size(), CV_8UC3, cv::Scalar::all(0));
	for (int row = 0; row < segment_image.rows; ++row) {
		for (int col = 0; col < segment_image.cols; ++col) {
			if (mask_image.data && !mask_image.at<unsigned char>(row,col)) continue;
			int segment = segment_image.at<int>(row, col);
			result.at<cv::Vec3b>(row, col) = segment_color_map[segment];
		}
	}
	return result;
}

void ObjectModeler::prepareSegmentColorMap(const cv::Mat& segment_image)
{
	for (int row = 0; row < segment_image.rows; ++row) {
		for (int col = 0; col < segment_image.cols; ++col) {
			int segment = segment_image.at<int>(row, col);
			if (segment_color_map.find(segment) == segment_color_map.end()) {
				segment_color_map[segment] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
			}
		}
	}
}

void ObjectModeler::prepareSegmentColorMap(int max_number)
{
	for (int n = 0; n <= max_number; ++n) {
		if (segment_color_map.find(n) == segment_color_map.end()) {
			segment_color_map[n] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
	}
}

void ObjectModeler::segmentByMerging(const FrameT& frame, 
		const cv::Mat& input_segmentation,
		cv::Mat& output_segmentation, 
		std::vector<int>& output_component_sizes, 
		std::vector<Eigen::Vector3f>& output_mean_normals)
{
	int rows = frame.object_cloud_ptr->height;
	int cols = frame.object_cloud_ptr->width;
	if (!input_segmentation.data) throw std::runtime_error ("Currently requires an input segmentation (could be all 0's)");
	if (input_segmentation.rows != rows || input_segmentation.cols != cols) throw std::runtime_error ("input_segmentation.rows != rows || input_segmentation.cols != cols");

	const float min_dot_product = cos(params.segments_max_angle * M_PI / 180);

	// at the beginning we need to map from the input names to the disjoint set name to form the initial disjoint set
	size_t initial_vector_size = pv_list.size();

	// the input names are sequential so we can use a vector (though this was just recently a pain in the ass...map?)
	std::vector<int> input_to_disjoint_set_vector(initial_vector_size, -1);
	std::map<int, int> disjoint_set_to_output_map; 

	// we need to initialize these from the input (along with the edges)
	DisjointSet disjoint_set(rows * cols);
	typedef std::map<int, Eigen::Vector3f> NormalMapT;
	NormalMapT disjoint_set_normal_map;
	std::vector<MergeEdge> edge_v;

	// right, down-left, down, down-right
	const static int deltas[] = {0,1, 1,-1, 1,0, 1,1};

	// create edges
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			// skip invalid points
			if (!frame.object_cloud_normal_mask.at<unsigned char>(row,col)) continue;

			int index = row * cols + col;
			const PointT& point = frame.object_cloud_ptr->at(index);
			const pcl::Normal& normal = frame.object_normal_cloud_ptr->at(index);

			// get the input component
			int input_component = input_segmentation.at<int>(row,col);
			if (input_component > 0) {
				// if we have an input component, need to map to disjoint set
				int previous_disjoint_set_for_component = input_to_disjoint_set_vector[input_component];
				if (previous_disjoint_set_for_component < 0) {
					// This is the first time we've seen this input component
					// Map the input component to this disjoint set
					input_to_disjoint_set_vector[input_component] = index;
					disjoint_set_to_output_map[index] = input_component;
					disjoint_set_normal_map[index] = normal.getNormalVector3fMap();
				}
				else {
					// Merge this point into the component
					// new_disjoint_set will often but not always be disjoint_set_for_component
					int new_disjoint_set = disjoint_set.connect(index, previous_disjoint_set_for_component);
					// add the normal
					// Note that we now have "old" disjoint sets in component_mean_normal_map and disjoint_set_to_output_map
					input_to_disjoint_set_vector[input_component] = new_disjoint_set;
					disjoint_set_to_output_map[new_disjoint_set] = input_component;
					disjoint_set_normal_map[new_disjoint_set] = disjoint_set_normal_map[previous_disjoint_set_for_component] + normal.getNormalVector3fMap();					
				}
			}
			else {
				// no input component
				disjoint_set_normal_map[index] = normal.getNormalVector3fMap();
			}			

			// go through "delta" neighbors
			for (int n = 0; n < 4; ++n) {
				int n_row = row + deltas[2*n];
				int n_col = col + deltas[2*n+1];
				
				// skip invalid neighbors
				if (n_row < 0 || n_row >= rows || n_col < 0 || n_col >= cols) continue;
				if (!frame.object_cloud_normal_mask.at<unsigned char>(n_row,n_col)) continue;

				// don't put an edge if already the same component (though it is safe to do so)
				if (input_component > 0 && input_segmentation.at<int>(n_row,n_col) == input_component) continue;

				int n_index = n_row * cols + n_col;
				const PointT & n_point = frame.object_cloud_ptr->at(n_index);
				const pcl::Normal& n_normal = frame.object_normal_cloud_ptr->at(n_index);

				// check distance
				//if ( (point.getVector3fMap() - n_point.getVector3fMap()).squaredNorm() > max_distance_squared) continue;
				if (abs(point.z - n_point.z) > params.max_depth_sigmas * Noise::simpleAxial(point.z)) continue;

				// use normal dot as weight?
				float dot_product = normal.getNormalVector3fMap().dot(n_normal.getNormalVector3fMap());

				edge_v.push_back(MergeEdge());
				MergeEdge& edge = edge_v.back();
				edge.a = index;
				edge.b = n_index;
				edge.w = 1-dot_product; // use 1-dot so sort ascending works
			}
		}
	}
	std::sort(edge_v.begin(), edge_v.end());

	/////////////////////////
	// need to normalize component_mean_normal_map
	for (NormalMapT::iterator iter = disjoint_set_normal_map.begin(); iter != disjoint_set_normal_map.end(); ++iter) {
		iter->second.normalize();
	}

	///////////////////////////////
	// merging
	// for each edge, if current normal dot product is ok, merge
	// edges will only be between valid points
	// can assume the normal map is complete
	for (std::vector<MergeEdge>::iterator iter = edge_v.begin(); iter != edge_v.end(); ++iter) {
		int a_set = disjoint_set.find(iter->a);
		int b_set = disjoint_set.find(iter->b);
		if (a_set == b_set) continue;
		// different sets, so check normal agreement:
		NormalMapT::iterator a_normal_find = disjoint_set_normal_map.find(a_set);
		NormalMapT::iterator b_normal_find = disjoint_set_normal_map.find(b_set);
		const Eigen::Vector3f& a_normal = a_normal_find->second;
		const Eigen::Vector3f& b_normal = b_normal_find->second;
		float dot_product = a_normal.dot(b_normal);
		if (dot_product < min_dot_product) continue;

		// needed to check and maintain input component mapping to output
		int a_output_component = disjoint_set_to_output_map[a_set];
		int b_output_component = disjoint_set_to_output_map[b_set];
		// if a and b are part of different input sets, don't merge
		if (a_output_component != 0 && b_output_component != 0) continue;

		// join:
		int a_size = disjoint_set.size(a_set);
		int b_size = disjoint_set.size(b_set);
		int new_set = disjoint_set.connect(a_set, b_set);
		disjoint_set_normal_map[new_set] = (a_size * a_normal + b_size * b_normal).normalized();

		// also update disjoint_set_to_output_map
		int new_set_output_component = disjoint_set_to_output_map[new_set];
		// if the new_set output component is zero, make sure we didn't lose our input component
		if (new_set_output_component == 0) {
			if (a_output_component != 0) disjoint_set_to_output_map[new_set] = a_output_component;
			else if (b_output_component != 0) disjoint_set_to_output_map[new_set] = b_output_component;
			// else we don't have an input component for either set
		}
	}

	////////////////////
	// merge in small segments at the end?
	for (std::vector<MergeEdge>::iterator iter = edge_v.begin(); iter != edge_v.end(); ++iter) {
		int a_set = disjoint_set.find(iter->a);
		int b_set = disjoint_set.find(iter->b);
		if (a_set == b_set) continue;
		// different sets, make sure both are large enough
		int a_size = disjoint_set.size(a_set);
		int b_size = disjoint_set.size(b_set);
		if (a_size >= params.segments_min_size && b_size >= params.segments_min_size) continue;
		
		// needed to check and maintain input component mapping to output
		int a_output_component = disjoint_set_to_output_map[a_set];
		int b_output_component = disjoint_set_to_output_map[b_set];
		// if a and b are part of different input sets, don't merge
		if (a_output_component != 0 && b_output_component != 0) continue;

		int new_set = disjoint_set.connect(a_set, b_set);
		NormalMapT::iterator a_normal_find = disjoint_set_normal_map.find(a_set);
		NormalMapT::iterator b_normal_find = disjoint_set_normal_map.find(b_set);
		const Eigen::Vector3f& a_normal = a_normal_find->second;
		const Eigen::Vector3f& b_normal = b_normal_find->second;
		disjoint_set_normal_map[new_set] = (a_size * a_normal + b_size * b_normal).normalized();

		// also update disjoint_set_to_output_map
		int new_set_output_component = disjoint_set_to_output_map[new_set];
		// if the new_set output component is zero, make sure we didn't lose our input component
		if (new_set_output_component == 0) {
			if (a_output_component != 0) disjoint_set_to_output_map[new_set] = a_output_component;
			else if (b_output_component != 0) disjoint_set_to_output_map[new_set] = b_output_component;
			// else we don't have an input component for either set
		}
	}


	//////////////////////
	// form output

	// need to extend input numbering potentially!!!
	output_segmentation = cv::Mat(rows, cols, CV_32SC1, cv::Scalar(0));
	output_component_sizes.assign(initial_vector_size, 0);
	output_mean_normals.assign(initial_vector_size, Eigen::Vector3f(0,0,0));

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			if (!frame.object_cloud_normal_mask.at<unsigned char>(row,col)) continue;

			int index = row * cols + col;
			int set = disjoint_set.find(index);

			std::map<int,int>::iterator find_component = disjoint_set_to_output_map.find(set);

			// figure out which component
			int component = 0;
			if (find_component == disjoint_set_to_output_map.end() || find_component->second == 0) {
				component = output_component_sizes.size();
				disjoint_set_to_output_map[set] = component;
				output_component_sizes.push_back(disjoint_set.size(set));
				output_mean_normals.push_back(disjoint_set_normal_map[set]);
			}
			else {
				component = find_component->second;
				// the first time we see an input component
				if (output_component_sizes[component] == 0) {
					output_component_sizes[component] = disjoint_set.size(set);
					output_mean_normals[component] = disjoint_set_normal_map[set];
				}
			}

			output_segmentation.at<int>(row,col) = component;
		}
	}
}

// I guess I'll put this here :)
bool operator<(const MergeEdge &a, const MergeEdge &b) {
  return a.w < b.w;
}


void ObjectModeler::getInitialSegmentation(const FrameT& frame, cv::Mat & result_render_segments, cv::Mat & result_consistent_segments)
{
	cv::Rect render_rect = frame.object_rect;
	result_render_segments = cv::Mat(render_rect.size(), CV_32S, cv::Scalar(0));
	result_consistent_segments = cv::Mat(render_rect.size(), CV_32S, cv::Scalar(0));

	bool is_first_frame = isEmpty();
	if (is_first_frame) {
		return;
	}

	// gotta render in order to have a "starter" segmentation
	Eigen::Vector2f render_f(params.camera_focal_x, params.camera_focal_y);
	Eigen::Vector2f render_c(params.camera_center_x, params.camera_center_y);
	RenderBuffers render_buffers(*cl_ptr);
	renderPatchVolumes(object_pose, render_f, render_c, params.camera_z_min, params.camera_z_max, render_rect, -1, params.pv_max_age_before_considered_loop, -1, false, false, render_buffers);

	// get clouds
	CloudT::Ptr render_points(new CloudT);
	pcl::PointCloud<pcl::Normal>::Ptr render_normals(new pcl::PointCloud<pcl::Normal>);
	extractRenderBuffersToClouds(render_rect, render_buffers, *render_points, *render_normals);

	// get segments
	render_buffers.readRenderMask((int*)result_render_segments.data);

	// debug
	if (params.segment_debug_images) {
		cv::Mat render_points_mat = cloudToImage(*render_points);
		cv::Mat render_normals_mat = normalCloudToImage(*render_normals, light_direction);
		showInWindow("getInitialSegmentation render", create1x2(render_points_mat, render_normals_mat));
	}

	// need to make sure render_segments are geometrically consistent with the new frame
	getConsistentRenderSegments(frame, *render_points, *render_normals, result_render_segments, result_consistent_segments);
}

CloudT::Ptr ObjectModeler::getCloudForSegment(FrameT const& frame, int c, cv::Mat const& segments) const
{
	CloudT::Ptr result(new CloudT);

	for (int row = 0; row < segments.rows; ++row) {
		for (int col = 0; col < segments.cols; ++col) {
			int this_c = segments.at<int>(row,col);
			if (this_c == c) {
				const PointT& p = frame.object_cloud_ptr->at(col,row);
				result->push_back(p);
			}
		}
	}

	return result;
}

void ObjectModeler::updateGrid(FrameT const& frame)
{
	///////////////////////////
	// rendering and alignment can stay the same!

	// loop over point, ensure allocation for cloud
	CloudT::Ptr cloud_in_grid(new CloudT);
	pcl::transformPointCloud(*frame.object_cloud_ptr, *cloud_in_grid, object_pose.inverse());

	std::set<int> pvs_touched;
	for (CloudT::iterator iter = cloud_in_grid->begin(); iter != cloud_in_grid->end(); ++iter) {
		const PointT & p = *iter;
		if (pcl_isnan(p.z)) continue;
		Eigen::Array3i grid_cell;
		Eigen::Array3f grid_cell_f = p.getArray3fMap() / params.volume_cell_size / params.pg_size;
		grid_cell[0] = (int)floor(grid_cell_f[0]);
		grid_cell[1] = (int)floor(grid_cell_f[1]);
		grid_cell[2] = (int)floor(grid_cell_f[2]);
		boost::tuple<int,int,int> key(grid_cell[0],grid_cell[1],grid_cell[2]);

		std::vector<int> & cells_for_spot = grid_to_list_map[key];
		int existing_active_pv = -1;
		for (std::vector<int>::iterator c_iter = cells_for_spot.begin(); c_iter != cells_for_spot.end(); ++c_iter) {
			int age_to_use = input_frame_counter - pv_list.at(*c_iter)->frame_last_in_frustum;
			if (params.pv_max_age_before_considered_loop >= 0 && age_to_use > params.pv_max_age_before_considered_loop) continue;
			existing_active_pv = *c_iter;
			break;
		}
		if (existing_active_pv < 0) {
			int cells_with_border = params.pg_size + 2 * params.pg_border;
			Eigen::Array3i cell_counts(cells_with_border, cells_with_border, cells_with_border);
			float cell_size = params.volume_cell_size;
			Eigen::Array3f cell_corner = grid_cell.cast<float>() * cell_size * params.pg_size - cell_size * params.pg_border;
			Eigen::Affine3f grid_pose;
			grid_pose = Eigen::Translation3f(cell_corner);
			pv_list.push_back(createNewPatchVolume(cell_counts + 1 /* "overlap" */, cell_size, grid_pose));
			int new_c = pv_list.size() - 1;
			cells_for_spot.push_back(new_c);
			existing_active_pv = new_c;
		}
		pvs_touched.insert(existing_active_pv);
	}

	rs_pvs_touched.push(pvs_touched.size());
	cout << "(grid) pvs_touched: " << rs_pvs_touched.summary() << endl;

	// DUPLICATE FROM UPDATEPATCHVOLUMES
	// create ImageBuffers so we only transfer to GPU once
	ImageBuffer buffer_depth_image(*cl_ptr);
	ImageBuffer buffer_color_image(*cl_ptr);
	ImageBuffer buffer_segments(*cl_ptr);
	cv::Mat object_depth_image(frame.image_depth.size(), frame.image_depth.type(), cv::Scalar(0));
	frame.image_depth.copyTo(object_depth_image, frame.object_mask);
	buffer_depth_image.writeFromBytePointer(object_depth_image.data, object_depth_image.total() * sizeof(float));
	cv::Mat image_color_uchar4; // make this once in prepare frame?
	cv::cvtColor(frame.image_color, image_color_uchar4, CV_BGR2BGRA);
	buffer_color_image.writeFromBytePointer(image_color_uchar4.data, image_color_uchar4.total() * 4 * sizeof(uint8_t));
	// end dup

	for (std::set<int>::iterator iter = pvs_touched.begin(); iter != pvs_touched.end(); ++iter) {
		// copied originally from addPointsToPatchVolumeWithBuffers
		ObjectModeler::PVStruct & pv_struct = *pv_list.at(*iter);
		pv_struct.tsdf_ptr->addFrame(object_pose * pv_struct.pose, buffer_depth_image, buffer_color_image, buffer_segments, 0);
	}

	// DUPLICATE
	// deallocate patch volumes
	if (params.pv_age_to_deallocate > 0) {
		for (int c = 1; c < pv_list.size(); ++c) {
			int age_to_use = input_frame_counter - pv_list.at(c)->frame_last_in_frustum;
			if (pv_list.at(c)->tsdf_ptr->buffersAreAllocated() && age_to_use >= params.pv_age_to_deallocate) {
				pv_list.at(c)->tsdf_ptr->deallocateVolumeBuffers();
			}
		}
	}

	// debug
	printPatchVolumeMemorySummary();

	// now update the graph / loop closure / etc
	bool ignore_did_optimize = updatePatchVolumeAndCameraPoseGraph(frame);

	// visualize
	updatePatchVolumeVisualizer();
}

void ObjectModeler::updatePatchVolumes(FrameT const& frame)
{
	// get initial segmentation (involves a render)
	cv::Rect render_rect = frame.object_rect;
	cv::Mat render_segments;
	cv::Mat consistent_segments;
	getInitialSegmentation(frame, render_segments, consistent_segments);

	// expand segmentation to all frame points
	cv::Mat segments;
	std::vector<int> segment_sizes;
	std::vector<Eigen::Vector3f> segment_mean_normals;
	segmentByMerging(frame, consistent_segments, segments, segment_sizes, segment_mean_normals);
	
	if (params.segment_debug_images) {
		cv::Mat segments_c = randomlyColorSegments(segments, frame.object_cloud_normal_mask);
		showInWindow("segments", segments_c);

		cv::Mat consistent_segments_c = randomlyColorSegments(consistent_segments, frame.object_cloud_normal_mask);
		showInWindow("consistent segments", consistent_segments_c);

		// also save
		if (params.save_segment_debug_images) {
			//cv::Mat normals_colored = normalCloudToRGBImage(*frame.object_normal_cloud_ptr);
			//cv::Mat display_depth = getPrettyDepthImage(frame.image_depth);
			cv::Mat frame_normals_lit = normalCloudToImage(*frame.object_normal_cloud_ptr, light_direction);

			savePNGWithIndex("segments", segments_c, input_frame_counter);

			// Disable each single for now...
			cout << "Disabled individual seg debug images" << endl;
#if 0
			savePNGWithIndex("segments-normals", normals_colored, input_frame_counter);
			savePNGWithIndex("segments-input-image", frame.image_color, input_frame_counter);
			savePNGWithIndex("segments-input-depth", display_depth, input_frame_counter);
#endif

			// also a 2x2 with
			std::vector<cv::Mat> v_images;
			v_images.push_back(frame.image_color);
			v_images.push_back(frame_normals_lit);
			v_images.push_back(consistent_segments_c);
			v_images.push_back(segments_c);
			cv::Mat all_four = createMxN(2,2,v_images);
			savePNGWithIndex("segments-all-four", all_four, input_frame_counter);
		}
	}

	// create ImageBuffers so we only transfer to GPU once
	ImageBuffer buffer_depth_image(*cl_ptr);
	ImageBuffer buffer_color_image(*cl_ptr);
	ImageBuffer buffer_segments(*cl_ptr);
	buffer_depth_image.writeFromBytePointer(frame.image_depth.data, frame.image_depth.total() * sizeof(float));
	cv::Mat image_color_uchar4; // make this once in prepare frame?
	cv::cvtColor(frame.image_color, image_color_uchar4, CV_BGR2BGRA);
	buffer_color_image.writeFromBytePointer(image_color_uchar4.data, image_color_uchar4.total() * 4 * sizeof(uint8_t));
	// segments are only object_rect sized
	cv::Mat segments_image_sized(frame.image_depth.size(), CV_32SC1, cv::Scalar(0));
	segments.copyTo(segments_image_sized(frame.object_rect));
	buffer_segments.writeFromBytePointer(segments_image_sized.data, segments_image_sized.total() * sizeof(int));

	int first_new_segment = pv_list.size();
	// put these in at the very end?
	std::vector<PVStructPtr> pv_list_from_splitting;
	std::vector<Eigen::Vector3f> segment_mean_normals_from_splitting;

	// expand old volumes
	pcl::StopWatch sw_expand;
	for (int c = 1; c < first_new_segment; ++c) {
		if (segment_sizes[c] == 0) {
			if (params.pv_verbose) {
				cout << "Skipping 0 size segment " << c << endl;
			}
			continue;
		}

		expandPatchVolumeToContainPoints(frame, c, segments, segment_mean_normals, *pv_list.at(c));
		addPointsToPatchVolumeWithBuffers(frame, c, segments, segment_sizes, segment_mean_normals, buffer_depth_image, buffer_color_image, buffer_segments, *pv_list.at(c));

		// now see if this patch volume is too big, and slice in half along the largest violator
		// smarter would be slicing along all, but results in 4 or 8 volumes, which annoys me
		if (params.pv_max_side_voxel_count > 0) {
			Eigen::Array3i origin = pv_list.at(c)->tsdf_ptr->getVolumeOrigin();
			Eigen::Array3i cell_counts = pv_list.at(c)->tsdf_ptr->getVolumeCellCounts();
			Eigen::Array3i::Index max_side_index;
			int max_side_length = cell_counts.maxCoeff(&max_side_index);

			if (max_side_length > params.pv_max_side_voxel_count) {
				if (params.pv_verbose) {
					cout << "max side length " << c << " is " << max_side_length << " at index " << max_side_index << endl;
				}

				PVStructPtr new_pv_struct(new PVStruct(*pv_list.at(c)));
				new_pv_struct->tsdf_ptr.reset(new OpenCLTSDF(pv_list.at(c)->tsdf_ptr->clone())); // yeah, sure...probably not too much extra copying in here

				// init to same
				Eigen::Array3i old_origin = origin;
				Eigen::Array3i old_cell_counts = cell_counts;
				Eigen::Array3i new_origin = origin;
				Eigen::Array3i new_cell_counts = cell_counts;

				// modify appropriate coord
				// want to duplicate exactly 1 slice of cells (vertices!) along boundary
				old_cell_counts[max_side_index] = old_cell_counts[max_side_index] / 2 + 1 + params.pv_split_voxel_overlap;
				new_cell_counts[max_side_index] = (new_cell_counts[max_side_index]+1) / 2 + params.pv_split_voxel_overlap;
				new_origin[max_side_index] = origin[max_side_index] + cell_counts[max_side_index] - new_cell_counts[max_side_index];

#if 0
				// debug remove
				cout << "Original: " << origin.transpose() << " " << cell_counts.transpose() << endl;
				cout << "old to: " << old_origin.transpose() << " " << old_cell_counts.transpose() << endl;
				cout << "new to: " << new_origin.transpose() << " " << new_cell_counts.transpose() << endl;
#endif

				pv_list.at(c)->tsdf_ptr->resize(old_origin, old_cell_counts);
				new_pv_struct->tsdf_ptr->resize(new_origin, new_cell_counts);

				// assume points divided in half
				pv_list.at(c)->points_added /= 2;
				new_pv_struct->points_added /= 2;

				// lock expand direction
				pv_list.at(c)->tsdf_ptr->max_expand_locked[max_side_index] = true;
				new_pv_struct->tsdf_ptr->min_expand_locked[max_side_index] = true;

				// note that no pixels in segments will have the int corresponding to this, so no points will be added later
				//pv_list.push_back(new_pv_struct);
				//segment_mean_normals.push_back(segment_mean_normals.at(c));
				//segment_sizes.push_back(0);
				// instead, don't need the normals or whatever...just add this at the end
				// note that it shares a vertex_id with the one it was split from
				pv_list_from_splitting.push_back(new_pv_struct);

				// can use this as a place to test merging
				if (params.pv_debug_add_volumes) {
					// do stuff
					cout << "pv_debug_add_volumes..." << endl;

#if 0
					// we reuse cell_counts
					OpenCLTSDFPtr test_tsdf_ptr;
					test_tsdf_ptr.reset(new OpenCLTSDF(*cl_ptr, *opencl_tsdf_kernels_ptr,
						params.camera_focal_x, params.camera_focal_y, params.camera_center_x, params.camera_center_y, params.camera_size_x, params.camera_size_y,
						cell_counts[0], cell_counts[1], cell_counts[2], params.volume_cell_size,
						params.volume_max_weight,
						params.volume_step_size,
						params.volume_normals_delta,
						params.volume_use_most_recent_color));

					test_tsdf_ptr->addVolume(*pv_list.at(c)->tsdf_ptr, Eigen::Affine3f::Identity());
					test_tsdf_ptr->addVolume(*new_pv_struct->tsdf_ptr, Eigen::Affine3f::Identity());

					// now make a mesh, or render or something...
					typedef pcl::PointCloud<pcl::PointXYZRGBNormal> VertexCloudT;
					VertexCloudT::Ptr vertex_cloud_ptr;
					boost::shared_ptr<std::vector<pcl::Vertices> > triangles_ptr;

					// use the same 
					// only conflicts if you press "m"
					boost::mutex::scoped_lock lock(mutex_pv_meshes_show);
					tm_pv_show_generated_list.clear();
					tc_pv_compare_to_mesh_list.clear();
					fs::path dump_path_debug = prepareDumpPath("pv_meshes_debug");

					generateMeshForTSDF(*test_tsdf_ptr, vertex_cloud_ptr, triangles_ptr);

					// don't bother checking for empty?
					//if (triangles_ptr->empty()) {

					// note we apply pv_list.at(c)-pose
					VertexCloudT::Ptr transformed_vertex_cloud_ptr(new VertexCloudT);
					pcl::transformPointCloud(*vertex_cloud_ptr, *transformed_vertex_cloud_ptr, pv_list.at(c)->original_pose);
					fs::path dump_filename = dump_path_debug / ((boost::format("debug-mesh-%05d-pv-%04d.ply") % output_frame_counter % c).str());
					saveMesh(dump_filename, transformed_vertex_cloud_ptr, triangles_ptr);

					pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr v(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
					pcl::transformPointCloud(*vertex_cloud_ptr, *v, object_pose * pv_list.at(c)->pose);
					//std::string mesh_name = (boost::format("pv_mesh_%05d")%c).str();
					std::string mesh_name = "debug-add-volume-mesh";
					tm_pv_show_generated_list.push_back(boost::shared_ptr<ToggleMesh<pcl::PointXYZRGBNormal> >());
					tm_pv_show_generated_list.back().reset(new ToggleMesh<pcl::PointXYZRGBNormal>(mesh_name, true));
					tm_pv_show_generated_list.back()->setCloud(v, triangles_ptr, 1.0);

					// show the voxels as well
					// yeah, this is slow:
					//std::string cloud_name = (boost::format("pv_cloud_%05d")%c).str();
					std::string cloud_name = "debug-add-volume-cloud";
					tc_pv_compare_to_mesh_list.push_back(boost::shared_ptr<ToggleCloud<PointT> >());
					tc_pv_compare_to_mesh_list.back().reset(new ToggleCloud<PointT>(cloud_name, true));
					CloudT::Ptr voxel_points = getVolumePointsFromOpenCLTSDFForPose(*test_tsdf_ptr, object_pose * pv_list.at(c)->pose, params.volume_debug_show_max_points);
					tc_pv_compare_to_mesh_list.back()->setCloud(voxel_points);
#endif
				}
			}
		}
	}

	cout << "[TIMING] Expand old patch volumes took: " << sw_expand.getTime() << " ms." << endl;

	// debug
	if (segment_sizes.size() != segment_mean_normals.size()) {
		cout << "YOU FAIL IT!" << endl;
		exit(1);
	}

	// the rest of the new ones
	for (int c = first_new_segment; c < segment_mean_normals.size(); ++c) {
		if (segment_sizes[c] == 0) {
			if (params.pv_verbose) {
				cout << "Skipping 0 size segment " << c << endl;
			}
			continue;
		}

		if (segment_sizes[c] < params.pv_min_size_to_create_new) {
			if (params.pv_verbose) {
				cout << "Not creating new PV with index " << c << " and size " << segment_sizes[c] << endl;
			}
			continue;
		}

		CloudT::Ptr segment_cloud = getCloudForSegment(frame, c, segments);

		if (params.pv_verbose) {
			cout << "Creating new PV with index " << c << " and size " << segment_cloud->size() << endl;
		}

		ObjectModeler::PVStructPtr new_pv_struct = createNewPatchVolumeFromCloud(segment_cloud, segment_mean_normals.at(c));
		addPointsToPatchVolumeWithBuffers(frame, c, segments, segment_sizes, segment_mean_normals, buffer_depth_image, buffer_color_image, buffer_segments, *new_pv_struct);
		pv_list.push_back(new_pv_struct);
	}

	// the ones from splitting at the end?
	for (int i = 0; i < pv_list_from_splitting.size(); ++i) {
		pv_list.push_back(pv_list_from_splitting.at(i));
	}

	// deallocate patch volumes
	if (params.pv_age_to_deallocate > 0) {
		for (int c = 1; c < pv_list.size(); ++c) {
			int age_to_use = input_frame_counter - pv_list.at(c)->frame_last_in_frustum;
			if (pv_list.at(c)->tsdf_ptr->buffersAreAllocated() && age_to_use >= params.pv_age_to_deallocate) {
				pv_list.at(c)->tsdf_ptr->deallocateVolumeBuffers();
			}
		}
	}

	// debug
	printPatchVolumeMemorySummary();

	// now update the graph / loop closure / etc
	bool ignore_did_optimize = updatePatchVolumeAndCameraPoseGraph(frame);

	updatePatchVolumeVisualizer();
}

size_t ObjectModeler::getAllocatedPatchVolumeSize()
{
	size_t total_size_bytes = 0;
	for (int c = 1; c < pv_list.size(); ++c) {
		size_t byte_size = pv_list.at(c)->tsdf_ptr->getAllocatedByteSize();
		size_t mb_size = byte_size / (1<<20);
		if (params.pv_verbose) {
			cout << "PV " << c << " allocated size: " << byte_size << " bytes = " << mb_size << " MB." << endl;
		}
		total_size_bytes += byte_size;
	}
	size_t allocated_mb = (total_size_bytes/ (1<<20));

	return allocated_mb;
}

size_t ObjectModeler::getRequiredPatchVolumesSize()
{
	size_t total_size_bytes = 0;
	for (int c = 1; c < pv_list.size(); ++c) {
		size_t byte_size = pv_list.at(c)->tsdf_ptr->getRequiredByteSize();
		size_t mb_size = byte_size / (1<<20);
		if (params.pv_verbose) {
			cout << "PV " << c << " required size: " << byte_size << " bytes = " << mb_size << " MB." << endl;
		}
		total_size_bytes += byte_size;
	}
	size_t required_mb = (total_size_bytes/ (1<<20));

	return required_mb;
}

void ObjectModeler::printPatchVolumeMemorySummary()
{
	size_t allocated = getAllocatedPatchVolumeSize();
	size_t required = getRequiredPatchVolumesSize();
	cout << "Patch volume count: " << pv_list.size() << endl;
	cout << "Allocated MB: " << allocated << endl;
	cout << "Required MB:  " << required << endl;
}

ObjectModeler::PVStructPtr ObjectModeler::createNewPatchVolumeFromCloud(const CloudT::Ptr & cloud_ptr, const Eigen::Vector3f & segment_normal)
{
	const static Eigen::Vector3f canonical_normal(0,0,-1);

	// compute centering transform
	// These transformations when applied to the segment cloud will center it at the origin, normal pointed to the camera
	Eigen::Affine3f centering_pose = Eigen::Affine3f::Identity();

	if (params.pv_use_covariance_to_create) {
		Eigen::Quaternionf rotation;

		Eigen::Vector4f centroid;
		Eigen::Matrix3f covariance;
		pcl::computeMeanAndCovarianceMatrix(*cloud_ptr, covariance, centroid);

		Eigen::Matrix3f eigen_vecs;
		Eigen::Vector3f eigen_vals;
		pcl::eigen33(covariance, eigen_vecs, eigen_vals);

		// simplest is just align the first dimension
		Eigen::Vector3f smallest_ev = eigen_vecs.row(0);
		// dupish
		Eigen::Vector3f axis = smallest_ev.cross(canonical_normal).normalized();
		float angle = acos(smallest_ev.dot(canonical_normal));
		static const float min_angle = 1e-5;
		if (angle > min_angle) {
			rotation = Eigen::AngleAxisf(angle, axis);
		}
		else {
			rotation = Eigen::Quaternionf(1,0,0,0); // identity (avoid NaN axis problem)
		}

		Eigen::Vector3f translate_centroid = -centroid.head<3>();

		// instead, transform cloud by rotation, determine bounds, then translate to center these
		Eigen::Affine3f aligning_pose;
		aligning_pose = rotation * Eigen::Translation3f(translate_centroid);
		CloudT::Ptr cloud_aa(new CloudT);
		pcl::transformPointCloud(*cloud_ptr, *cloud_aa, aligning_pose);
		Eigen::Array3f bb_min;
		Eigen::Array3f bb_max;
		getBoundingBox(cloud_aa, bb_min, bb_max);
		Eigen::Array3f bb_center = (bb_min + bb_max) / 2;
		Eigen::Vector3f translate_final = -bb_center.matrix();

		centering_pose = Eigen::Translation3f(translate_final) * aligning_pose;
	}
	else {
		Eigen::Quaternionf rotation;
		Eigen::Vector3f translation;

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*cloud_ptr, centroid);
		translation = -centroid.head<3>();

		Eigen::Vector3f axis = segment_normal.cross(canonical_normal).normalized();
		float angle = acos(segment_normal.dot(canonical_normal));
		static const float min_angle = 1e-5;
		if (angle > min_angle) {
			rotation = Eigen::AngleAxisf(angle, axis);
		}
		else {
			rotation = Eigen::Quaternionf(1,0,0,0); // identity (avoid NaN axis problem)
		}

		centering_pose = rotation * Eigen::Translation3f(translation);
	}

	CloudT::Ptr cloud_in_box(new CloudT);
	pcl::transformPointCloud(*cloud_ptr, *cloud_in_box, centering_pose);

	// get centered cell count
	Eigen::Array3f bb_min;
	Eigen::Array3f bb_max;
	getBoundingBox(cloud_in_box, bb_min, bb_max);
	Eigen::Array3f max_abs_bb = bb_min.abs().max(bb_max.abs());
	Eigen::Array3f volume_size (2 * (max_abs_bb + params.pv_initial_border_size));
	Eigen::Array3i cell_counts = (((volume_size/params.volume_cell_size).cast<int>()+2)/2*2);  // next even number

	float cell_size = params.volume_cell_size; // refer multiple times

	// get pose of patch relative to object pose
	Eigen::Translation3f centering_translation(-cell_counts.cast<float>() * cell_size / 2);
	Eigen::Affine3f pv_pose = object_pose.inverse() * centering_pose.inverse() * centering_translation;

	return createNewPatchVolume(cell_counts, cell_size, pv_pose);
}

ObjectModeler::PVStructPtr ObjectModeler::createNewPatchVolume(Eigen::Array3i const& cell_counts, float cell_size, Eigen::Affine3f const& pose)
{
	PVStructPtr result(new PVStruct);
	result->tsdf_ptr.reset(new OpenCLTSDF(*cl_ptr, *all_kernels_ptr->getKernelsBuilderTSDF(),
		params.camera_focal_x, params.camera_focal_y, params.camera_center_x, params.camera_center_y, params.camera_size_x, params.camera_size_y,
		cell_counts[0], cell_counts[1], cell_counts[2], cell_size,
		//-cell_counts[0]/2, -cell_counts[1]/2, -cell_counts[2]/2,
		0,0,0,
		params.volume_max_weight));
	result->pose.setIdentity();
	result->frame_last_in_frustum = input_frame_counter;
	result->vertex_id = -1;
	result->normal = Eigen::Vector3f(0,0,-1);
	result->points_added = 0;
	result->pose = pose;
	result->original_pose = result->pose;

	return result;
}

void ObjectModeler::getBoundingBox(CloudT::Ptr const& cloud_ptr, Eigen::Array3f & bb_min_result, Eigen::Array3f & bb_max_result) const
{
	Eigen::Vector4f bb_min;
	Eigen::Vector4f bb_max;
	pcl::getMinMax3D(*cloud_ptr, bb_min, bb_max);
	bb_min_result = bb_min.array().head<3>();
	bb_max_result = bb_max.array().head<3>();
}

void ObjectModeler::expandPatchVolumeToContainPoints(FrameT const& frame, int c, cv::Mat const& segments, std::vector<Eigen::Vector3f> const& segment_normals, ObjectModeler::PVStruct & pv_struct)
{
	CloudT::Ptr cloud = getCloudForSegment(frame, c, segments);
	if (cloud->empty()) return; 

	Eigen::Affine3f cloud_to_volume = (object_pose * pv_struct.pose).inverse();
	CloudT::Ptr cloud_in_box(new CloudT);
	pcl::transformPointCloud(*cloud, *cloud_in_box, cloud_to_volume);

	float cell_size = pv_struct.tsdf_ptr->getVolumeCellSize();
	int expand_border_cells = params.pv_expand_border_size / cell_size;

	// bb
	Eigen::Array3f bb_min, bb_max;
	getBoundingBox(cloud_in_box, bb_min, bb_max);

	// in cells:
	Eigen::Array3f bb_min_cells_f = bb_min / cell_size - 0.5;
	Eigen::Array3f bb_max_cells_f = bb_max / cell_size + 0.5;
	Eigen::Array3i bb_min_cells( (int)bb_min_cells_f[0], (int)bb_min_cells_f[1], (int)bb_min_cells_f[2] ); 
	Eigen::Array3i bb_max_cells( (int)bb_max_cells_f[0], (int)bb_max_cells_f[1], (int)bb_max_cells_f[2] ); 

	// get current cell dimensions
	Eigen::Array3i current_bb_min_cells = pv_struct.tsdf_ptr->getVolumeOrigin();
	Eigen::Array3i current_bb_max_cells = current_bb_min_cells + pv_struct.tsdf_ptr->getVolumeCellCounts();

	// proposal
	Eigen::Array3i proposed_min_cells = bb_min_cells.min(current_bb_min_cells);
	Eigen::Array3i proposed_max_cells = bb_max_cells.max(current_bb_max_cells);

	for (int i = 0; i < 3; ++i) {
		if (!pv_struct.tsdf_ptr->max_expand_locked[i]) {
			if (proposed_max_cells[i] > current_bb_max_cells[i]) proposed_max_cells[i] += expand_border_cells;
		}
		else {
			proposed_max_cells[i] = current_bb_max_cells[i];
		}

		if (!pv_struct.tsdf_ptr->min_expand_locked[i]) {
			if (proposed_min_cells[i] < current_bb_min_cells[i]) proposed_min_cells[i] -= expand_border_cells;
		}
		else {
			proposed_min_cells[i] = current_bb_min_cells[i];
		}
	}

	if ( (proposed_max_cells > current_bb_max_cells).any() || (proposed_min_cells < current_bb_min_cells).any()) {
		// This MUST match what you did above
		// also must be either all expand or all contract (for some reason I can't figure out right now)
		Eigen::Array3i new_origin = proposed_min_cells;
		Eigen::Array3i new_cell_counts = proposed_max_cells - proposed_min_cells;
		pv_struct.tsdf_ptr->resize(new_origin, new_cell_counts);
	}
}

void ObjectModeler::addPointsToPatchVolumeWithBuffers(const FrameT& frame, int c, cv::Mat const& segments, std::vector<int> const& segment_sizes, std::vector<Eigen::Vector3f> const& segment_normals, 
	ImageBuffer const& buffer_depth_image, ImageBuffer const& buffer_color_image, ImageBuffer const& buffer_segments, ObjectModeler::PVStruct & pv_struct) const
{
	int segment_size = segment_sizes[c];
	if (segment_size == 0) return;

	pv_struct.tsdf_ptr->addFrame(object_pose * pv_struct.pose, buffer_depth_image, buffer_color_image, buffer_segments, c);

	////////////////////
	// also update normal
	// transform patch normal to camera frame, compute new normal, then back to patch frame
	Eigen::Vector3f pv_normal_camera_frame = (object_pose * pv_struct.pose).linear() * pv_struct.normal;
	Eigen::Vector3f new_normal_camera_frame = (segment_size * segment_normals.at(c) + pv_struct.points_added * pv_normal_camera_frame).normalized();
	pv_struct.normal = (object_pose * pv_struct.pose).inverse().linear() * new_normal_camera_frame;
	pv_struct.points_added += segment_size;
}

CloudT::Ptr ObjectModeler::getLinesForAllPatchVolumes()
{
	CloudT::Ptr all_patch_lines(new CloudT);
	for (int c = 1; c < pv_list.size(); ++c) {
		CloudT::Ptr lines = getLinesForPatchVolume(c);
		*all_patch_lines += *lines;
	}
	return all_patch_lines;
}

CloudT::Ptr ObjectModeler::getLinesForPatchVolume(int c)
{
	std::vector<Eigen::Vector3f> corners = pv_list.at(c)->tsdf_ptr->getVolumeCorners(object_pose * pv_list.at(c)->pose);
	CloudT::Ptr lines = getLineSetEdges(corners);
	// color points
	//cv::Vec3b segment_color = segment_color_map[c];
	cv::Vec3b color = getPVStatusColor(c);
	for (int p = 0; p < lines->size(); ++p) {
		lines->at(p).r = color[2];
		lines->at(p).g = color[1];
		lines->at(p).b = color[0];
	}
	return lines;
}

void ObjectModeler::getConsistentRenderSegments(FrameT const& frame, CloudT const& render_cloud, pcl::PointCloud<pcl::Normal> const& render_normal_cloud, cv::Mat const& render_segments, cv::Mat& result_segments)
{
	result_segments = cv::Mat(render_segments.size(), render_segments.type(), cv::Scalar(0));

	int rows = render_segments.rows;
	int cols = render_segments.cols;

	const float min_dot_product = cos(params.segments_max_angle * M_PI / 180);

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			bool is_consistent = true;
			is_consistent = is_consistent && render_segments.at<int>(row,col) != 0;
			is_consistent = is_consistent && frame.object_cloud_normal_mask.at<unsigned char>(row,col);

			// check depth diff
			float frame_z = frame.object_cloud_ptr->at(col,row).z;
			is_consistent = is_consistent && abs(render_cloud.at(col,row).z - frame_z) <= params.max_depth_sigmas * Noise::simpleAxial(frame_z);

			// check normal diff
			Eigen::Vector3f const& normal_render = render_normal_cloud.at(col,row).getNormalVector3fMap();
			Eigen::Vector3f const& normal_frame = frame.object_normal_cloud_ptr->at(col,row).getNormalVector3fMap();
			is_consistent = is_consistent && normal_render.dot(normal_frame) >= min_dot_product;

			if (is_consistent) {
				result_segments.at<int>(row,col) = render_segments.at<int>(row,col);
			}
		}
	}
}

bool ObjectModeler::updatePatchVolumeAndCameraPoseGraph(const FrameT& frame)
{
	bool result = false; // did optimize?
	bool is_first_frame = (!pv_pose_graph_ptr);
	if (is_first_frame) {
		pv_pose_graph_ptr.reset(new G2OPoseGraph);
		pv_pose_graph_ptr->setVerbose(params.pv_pose_graph_verbose);
	}

	// start by adding new camera vertex
	CameraPoseStructPtr camera(new CameraPoseStruct);
	camera->pose = object_pose.inverse();
	camera->vertex_id = pv_pose_graph_ptr->addVertex(EigenUtilities::getIsometry3d(camera->pose), is_first_frame);
	vertex_to_camera_map[camera->vertex_id] = camera;
	camera_list.push_back(camera);

	// need to add vertices for new volumes
	for (int c = 1; c < pv_list.size(); ++c) {
		if (pv_list.at(c)->vertex_id < 0) {
			Eigen::Affine3f global_pose = pv_list.at(c)->pose;
			pv_list.at(c)->vertex_id = pv_pose_graph_ptr->addVertex(EigenUtilities::getIsometry3d(global_pose), false);
			vertex_to_pv_map[pv_list.at(c)->vertex_id] = pv_list.at(c);
		}
	}

	// having now updated the patch volumes, can tie together all patch volumes with an age of 0
	std::vector<int> updated_this_frame;
	for (int c = 1; c < pv_list.size(); ++c) {
		if (pv_list.at(c)->frame_last_in_frustum == input_frame_counter) {
			updated_this_frame.push_back(c);
		}
	}

	// now need to add relative edges from camera to "updated_this_frame"
	size_t edges_added = 0;
	for (std::vector<int>::iterator iter_tf = updated_this_frame.begin(); iter_tf != updated_this_frame.end(); ++iter_tf) {
		Eigen::Affine3f relative_pose = camera->pose.inverse() * pv_list.at(*iter_tf)->pose;
		bool did_add = pv_pose_graph_ptr->addEdge(camera->vertex_id, pv_list.at(*iter_tf)->vertex_id, EigenUtilities::getIsometry3d(relative_pose));
		if (did_add) edges_added++;
	}

	cout << "Added " << edges_added << " same-frame edges to pv pose graph." << endl;
	cout << "Pose graph has " << pv_pose_graph_ptr->getVertexCount() << " nodes and " << pv_pose_graph_ptr->getEdgeCount() << " edges." << endl;


	/////////////
	// loop closure
	if (params.pv_loop_closure) {
		// for debugging
		CloudICPTargetT::Ptr last_render_cloud(new CloudICPTargetT); // before, loop, optimize comparision
		Eigen::Affine3f pose_for_last_render_cloud;
		Eigen::Affine3f object_pose_before_loop_closure = object_pose; // should get set later
		Eigen::Affine3f object_pose_after_loop_closure = object_pose; // should get set later

		bool closure_found = false;
		static int pv_loop_frame_skip_counter = params.pv_loop_frame_skip;
		if (!is_first_frame && pv_loop_frame_skip_counter-- <= 0) {
			if (params.pv_verbose) {
				cout << "checking for loop closure..." << endl;
			}
			pv_loop_frame_skip_counter = params.pv_loop_frame_skip;

			// get pose relative to "old" patch volumes
			Eigen::Affine3f object_pose_according_to_closure = object_pose;
			std::set<int> old_pvs_used; // which "old" segments were aligned against
			object_pose_before_loop_closure = object_pose; // debug
			// old_pvs_used only set on success
			closure_found = checkForLoopClosure(frame, object_pose_according_to_closure, old_pvs_used, last_render_cloud, pose_for_last_render_cloud);
			if (closure_found) {
				loop_closure_count++;

				Eigen::Affine3f relative_pose = object_pose_according_to_closure * object_pose.inverse();

				cout << "Loop closure ICP success" << endl;
				cout << "relative_pose:\n" << relative_pose.matrix() << endl;
				cout << "object_pose_according_to_closure:\n" << object_pose_according_to_closure.matrix() << endl;

				int loop_edges_added = 0;

				// add edges between old pvs used and the new camera
				for (std::set<int>::iterator iter = old_pvs_used.begin(); iter != old_pvs_used.end(); ++iter) {
					Eigen::Affine3f corrected_old_pose = object_pose.inverse() * relative_pose * object_pose * pv_list.at(*iter)->pose;
					Eigen::Affine3f relative_pose = camera->pose.inverse() * corrected_old_pose;
					bool did_add = pv_pose_graph_ptr->addEdge(camera->vertex_id, pv_list.at(*iter)->vertex_id, EigenUtilities::getIsometry3d(relative_pose), params.pv_edge_loop_information_factor);
					if (did_add) {
						loop_edges_added++;
						edge_is_loop_set.insert(std::make_pair(camera->vertex_id, pv_list.at(*iter)->vertex_id));
					}
					else {
						cout << "--- WARNING: addEdge failed.  Camera id: " << camera->vertex_id << " Vertex id: " <<  pv_list.at(*iter)->vertex_id << endl;
					}
				}

				// debug
				cout << "added " << loop_edges_added << " loop edges" << endl;
				cout << "loop_closure_count: " << loop_closure_count << endl;

				object_pose = object_pose_according_to_closure;
				object_pose_after_loop_closure = object_pose; // debug
			}
			else {
				cout << "Loop closure ICP failure" << endl;
			}
		}

		if (closure_found || params.pv_debug_optimize_always) {
			optimizePoseGraph();
			result = true;
		}

		// can now do debug comparison if "closure found"
		if (closure_found && params.pv_loop_debug_images) {
			// so stupid...build up the stuff that's in icp_functor
			// hopefully match...sometime code this correctly
			Eigen::Vector2f f_vec(params.camera_focal_x, params.camera_focal_y);
			Eigen::Vector2f c_vec(params.camera_center_x, params.camera_center_y);
			Eigen::Vector2f offset_vec(frame.object_rect.x, frame.object_rect.y);

			//cv::Mat last_render_mat = cloudToImage(*projectRenderCloud(*last_render_cloud, f_vec, c_vec, offset_vec));

			CloudICPTargetT::Ptr last_render_cloud_before_closure(new CloudICPTargetT);
			pcl::transformPointCloud(*last_render_cloud, *last_render_cloud_before_closure, object_pose_before_loop_closure * pose_for_last_render_cloud.inverse());
			cv::Mat last_render_before_closure_mat = cloudToImage(*projectRenderCloud(*last_render_cloud_before_closure, f_vec, c_vec, offset_vec));

			CloudICPTargetT::Ptr last_render_cloud_after_closure(new CloudICPTargetT);
			pcl::transformPointCloud(*last_render_cloud, *last_render_cloud_after_closure, object_pose_after_loop_closure * pose_for_last_render_cloud.inverse());
			cv::Mat last_render_after_closure_mat = cloudToImage(*projectRenderCloud(*last_render_cloud_after_closure, f_vec, c_vec, offset_vec));

			CloudICPTargetT::Ptr last_render_cloud_after_optimize(new CloudICPTargetT);
			pcl::transformPointCloud(*last_render_cloud, *last_render_cloud_after_optimize, object_pose * pose_for_last_render_cloud.inverse());
			cv::Mat last_render_after_optimize = cloudToImage(*projectRenderCloud(*last_render_cloud_after_optimize, f_vec, c_vec, offset_vec));

			// also compare to frame
			cv::Mat frame_object_mat;
			frame.image_color(frame.object_rect).copyTo(frame_object_mat);

			cv::Mat before_closure_both = frame_object_mat * 0.5 + last_render_before_closure_mat * 0.5;
			cv::Mat after_closure_both = frame_object_mat * 0.5 + last_render_after_closure_mat * 0.5;
			cv::Mat after_optimize_both = frame_object_mat * 0.5 + last_render_after_optimize * 0.5;

			std::vector<cv::Mat> v_images;
			v_images.push_back(last_render_before_closure_mat);
			v_images.push_back(last_render_after_closure_mat);
			v_images.push_back(last_render_after_optimize);

			v_images.push_back(before_closure_both);
			v_images.push_back(after_closure_both);
			v_images.push_back(after_optimize_both);

			cv::Mat all_images = createMxN(2, 3, v_images);
			float scale = params.combined_debug_images_scale;
			cv::Mat all_images_scaled;
			cv::resize(all_images, all_images_scaled, cv::Size(), scale, scale, cv::INTER_NEAREST); 
			showInWindow("pv loop closure debug images", all_images_scaled);

			// also save full size
			savePNGWithIndex("loop-debug-all", all_images, input_frame_counter);

			// and save before / after 1x2
			cv::Mat simple_before_after = create1x2(before_closure_both, after_closure_both);
			savePNGWithIndex("loop-debug-simple", simple_before_after, input_frame_counter);
		}


		// here add keyframe?
		if (params.pv_loop_features) {
			addKeyframe(frame, updated_this_frame);
		}
	} // loop closure

	return result;
}

bool ObjectModeler::checkForLoopClosure(FrameT const& frame, Eigen::Affine3f & result_object_pose, std::set<int> & result_old_pvs_used, CloudICPTargetT::Ptr & last_render_cloud, Eigen::Affine3f & pose_for_last_render_cloud)
{
	bool closure_found = false;
	result_object_pose = object_pose;
	result_old_pvs_used.clear();
	if (!input_loop_poses.empty()) {
		int pose_index = input_frame_counter - 1;
		closure_found = input_loop_poses.at(pose_index).first;
		if (closure_found) {
			result_object_pose = input_loop_poses.at(pose_index).second;
			result_old_pvs_used.insert(input_loop_which_pvs.at(pose_index).begin(), input_loop_which_pvs.at(pose_index).end());
		}
	}
	else {
		// here do feature based check?
		// ok, maybe you should pull the render out?  or you at least need to decide on which patch volumes are in view, collect keyframes which saw them, and align against those

		RenderBuffers render_buffers(*cl_ptr);
		cv::Rect render_rect;
		Eigen::Affine3f initial_pose = object_pose;
		bool render_ok = renderForAlign(frame, true, initial_pose, render_buffers, render_rect);
		if (render_ok) {
			// need which_segment to find keyframes
			cv::Mat which_segment = cv::Mat(render_rect.height, render_rect.width, CV_32S);
			render_buffers.readRenderMask((int*)which_segment.data);

			// used to only set this if closure_found
			int* pixel = (int*)which_segment.data;
			for(int i = 0; i < which_segment.total(); ++i) {
				int s = *pixel++;
				if (s > 0) result_old_pvs_used.insert(s);
			}

			Eigen::Affine3f initial_relative_pose = Eigen::Affine3f::Identity();
			// only consider keyframes which have seen these pvs
			if (params.pv_loop_features) {
				std::set<int> keyframes_that_saw_pvs;
				for (std::set<int>::iterator iter = result_old_pvs_used.begin(); iter != result_old_pvs_used.end(); ++iter) {
					keyframes_that_saw_pvs.insert(pv_to_keyframes_map[*iter].begin(), pv_to_keyframes_map[*iter].end());
				}

				// check these keyframes poses against current pose
				Eigen::Affine3f frame_pose = object_pose.inverse();
				std::set<int> nearby_keyframes;
				for (std::set<int>::iterator iter = keyframes_that_saw_pvs.begin(); iter != keyframes_that_saw_pvs.end(); ++iter) {
					Eigen::Affine3f keyframe_pose = vertex_to_camera_map[keyframes.at(*iter).vertex_id]->pose;
					Eigen::Affine3f relative_pose = frame_pose * keyframe_pose.inverse(); // right?
					float translation = relative_pose.translation().norm();
					Eigen::AngleAxisf aa(relative_pose.rotation());
					float angle_degrees = 180.f / M_PI * aa.angle();
					if (translation < params.pv_keyframe_max_distance_match && angle_degrees < params.pv_keyframe_max_angle_match) {
						nearby_keyframes.insert(*iter);
					}
				}

				if (params.pv_keyframe_debug) {
					for (std::set<int>::iterator iter = nearby_keyframes.begin(); iter != nearby_keyframes.end(); ++iter) { 
						showInWindow("potential keyframe", keyframes.at(*iter).frame_ptr->image_color);
						cout << "potential keyframe: " << *iter << endl;
						processWaitKey();
					}
				}

				// attempt feature-based alignment against nearby keyframes.
				// set will be in order, so first success is oldest and the one we want
				for (std::set<int>::iterator iter = nearby_keyframes.begin(); iter != nearby_keyframes.end(); ++iter) {
					Eigen::Affine3f feature_result_pose;
					std::vector<cv::DMatch> inlier_matches;
					Eigen::Affine3f other_frame_pose = vertex_to_camera_map[keyframes.at(*iter).vertex_id]->pose;
					//bool feature_success = alignWithFeaturesToModel(other_frame_pose, frame, keyframes.at(*iter).frame_ptr->object_kp, feature_result_pose, inlier_matches);
					bool feature_success = alignWithFeaturesToModel(Eigen::Affine3f::Identity(), frame, keyframes.at(*iter).frame_ptr->object_kp, feature_result_pose, inlier_matches);
					if (feature_success) {
						cout << "Successful match against keyframe " << *iter << " with inlier_matches size: " << inlier_matches.size() << endl;

						// think in terms of "object pose"
						Eigen::Affine3f object_pose_from_features = feature_result_pose * other_frame_pose.inverse();
						initial_relative_pose = object_pose_from_features * initial_pose.inverse();

						break;
					}
				}
			}


			bool show_debug_images = params.pv_loop_debug_images;
			closure_found = alignWithCombinedOptimizationNew(frame, render_buffers, render_rect, show_debug_images, initial_pose, initial_relative_pose, result_object_pose);

			if (closure_found) {
				if (last_render_cloud) {
					last_render_cloud = renderBuffersToCloud(render_buffers, render_rect);
					pose_for_last_render_cloud = initial_pose;
				}
			}
		}
	}
	return closure_found;
}

void ObjectModeler::optimizePoseGraph()
{
	pcl::StopWatch sw;
	pv_pose_graph_ptr->optimize(params.pv_pose_graph_iterations);
	cout << "Ran this many iterations of pose graph optimization: " << params.pv_pose_graph_iterations << endl;
	// better update those poses after you optimize!
	for (int c = 1; c < pv_list.size(); ++c) {
		pv_list.at(c)->pose = EigenUtilities::getAffine3f(pv_pose_graph_ptr->getVertexPose(pv_list.at(c)->vertex_id));
	}
	// also updated camera poses
	for (int cam = 0; cam < camera_list.size(); ++cam) {
		camera_list.at(cam)->pose = EigenUtilities::getAffine3f(pv_pose_graph_ptr->getVertexPose(camera_list.at(cam)->vertex_id));
	}
	// and object pose?
	object_pose = camera_list.back()->pose.inverse();

	if (params.save_poses) {
		if (any_failures_save_exception) throw std::runtime_error("Can't save loop poses when any failures have happened (lazy Peter)");
		// save all object and camera poses
		fs::path dump_path = prepareDumpPath();
		fs::path object_poses_filename = dump_path / "object_poses_loop.txt";
		fs::path camera_poses_filename = dump_path / "camera_poses_loop.txt";
		std::ofstream ofs_object_pose_loop;
		std::ofstream ofs_camera_pose_loop;
		ofs_object_pose_loop.open(object_poses_filename.string().c_str());
		ofs_camera_pose_loop.open(camera_poses_filename.string().c_str());
		for (int i = 0; i < camera_list.size(); ++i) {
			ofs_object_pose_loop << EigenUtilities::transformToString(camera_list.at(i)->pose.inverse()) << endl;
			ofs_camera_pose_loop << EigenUtilities::transformToString(camera_list.at(i)->pose) << endl;
		}
	}
	rs_optimize_pose_graph.push(sw.getTime());
	cout << "[TIMING] optimizePoseGraph: " << rs_optimize_pose_graph.summary() << endl;
}

void ObjectModeler::globalStopWatchMark(const std::string& s)
{
	float ms = g_sw.getTime();
	cout << "[TIMING GLOBAL] " << s << ": " << ms << "ms." << endl;
	g_sw.reset();
}

void ObjectModeler::showPVGraphEdges(Eigen::Affine3f const& pose)
{
	if (pv_pose_graph_ptr) {
		pcl::StopWatch sw;

		CloudT::Ptr cloud(new CloudT);
		std::vector<std::pair<size_t, size_t> > edges;
		pv_pose_graph_ptr->getEdges(edges);
		if (params.pv_debug_print_edges) {
			cout << "Sorting edges...maybe slow?" << endl;
			std::sort(edges.begin(), edges.end());
		}
		cloud->resize(edges.size() * 2);
		for (size_t i = 0; i < edges.size(); ++i) {
			Eigen::Isometry3d pose1, pose2;
			int v1 = edges.at(i).first;
			int v2 = edges.at(i).second;
			pv_pose_graph_ptr->getVertexPose(v1, pose1);
			pv_pose_graph_ptr->getVertexPose(v2, pose2);
			PointT& p1 = cloud->at(2*i);
			PointT& p2 = cloud->at(2*i+1);
			p1.getVector3fMap() = (pose * pose1.cast<float>()).translation().cast<float>();
			p2.getVector3fMap() = (pose * pose2.cast<float>()).translation().cast<float>();

			if (params.pv_debug_print_edges) {
				cout << "Edge: " << v1 << " - " << v2 << endl;
			}

			if (edge_is_loop_set.find(std::make_pair(edges.at(i).first, edges.at(i).second)) != edge_is_loop_set.end()) {
				// loop edge
				p1.r = p2.r = 0;
				p1.g = p2.g = 255;
				p1.b = p2.b = 0;
			}
			else {
				// normal edge
				p1.r = p2.r = 255;
				p1.g = p2.g = 0;
				p1.b = p2.b = 0;
			}
		}
		tls_graph_edges.setLineSet(cloud);

		cout << "[TIMING] showPVGraphEdges: " << sw.getTime() << " ms." << endl;
	}
}

cv::Vec3b ObjectModeler::getPVStatusColor(int c)
{
	cv::Vec3b color;
	cv::Vec3b color_for_deallocated(255,255,255);
	if (params.white_background) {
		color_for_deallocated = cv::Vec3b(0,0,0);
	}
	static const cv::Vec3b color_for_old(100,100,100);
	int age_to_use = input_frame_counter - pv_list.at(c)->frame_last_in_frustum;
	if (pv_list.at(c)->tsdf_ptr->buffersAreAllocated() || params.pv_color_all_boxes) {
		color = segment_color_map[c];
	}
	else if (age_to_use <= params.pv_max_age_before_considered_loop) {
		color = color_for_deallocated;
	}
	else {
		color = color_for_old;
	}
	return color;
}

void ObjectModeler::showPVVolumeMesh()
{
	const uchar a = params.pv_mesh_alpha * 255; // alpha
	// build these up:
	boost::shared_ptr<std::vector<pcl::Vertices> > mesh_v(new std::vector<pcl::Vertices>);
	CloudT::Ptr mesh_c(new CloudT);

	for (size_t c = 1; c < pv_list.size(); ++c) {
		boost::shared_ptr<std::vector<pcl::Vertices> > v = getMeshVerticesForCorners(8 * (c - 1));
		std::vector<Eigen::Vector3f> corners = pv_list.at(c)->tsdf_ptr->getVolumeCorners(object_pose * pv_list.at(c)->pose);
		cv::Vec3b color = getPVStatusColor(c);
		CloudT::Ptr corner_cloud(new CloudT);
		corner_cloud->points.resize(corners.size());
		for (size_t i = 0; i < corners.size(); ++i) {
			corner_cloud->at(i).getVector3fMap() = corners.at(i);
			corner_cloud->at(i).r = color[2];
			corner_cloud->at(i).g = color[1];
			corner_cloud->at(i).b = color[0];
			corner_cloud->at(i).a = a;
		}

		// extend:
		mesh_v->insert(mesh_v->end(), v->begin(), v->end());
		mesh_c->insert(mesh_c->begin(), corner_cloud->begin(), corner_cloud->end());
	}

	tm_patch_volume_debug.setCloud(mesh_c, mesh_v, params.pv_mesh_alpha);
}

void ObjectModeler::showPVVolumeNormals()
{
	CloudT::Ptr line_c(new CloudT);
	const static float line_length = 0.1;
	for (size_t c = 1; c < pv_list.size(); ++c) {
		PointT p_center;
		PointT p_normal;
		p_center.r = p_normal.r = 255;
		p_center.g = p_normal.g = 0;
		p_center.b = p_normal.b = 255;
		p_center.getVector3fMap() = object_pose * pv_list.at(c)->pose * Eigen::Vector3f(0,0,0);
		Eigen::Vector3f normal_transformed = (object_pose * pv_list.at(c)->pose).linear() * pv_list.at(c)->normal;
		p_normal.getVector3fMap() = p_center.getVector3fMap() + line_length * normal_transformed;
		line_c->push_back(p_center);
		line_c->push_back(p_normal);
	}

	tls_patch_volume_normals.setLineSet(line_c);
}

void ObjectModeler::updatePatchVolumeVisualizer()
{
	// TODO: refactor the checks so they're inside the functions...
	if (!params.disable_cloud_viewer) {
		prepareSegmentColorMap(pv_list.size());

		pcl::StopWatch sw;
		if (params.pv_show_volume_edges) {
			tls_patch_volume_debug.setLineSet(getLinesForAllPatchVolumes());
		}
		if (params.pv_show_volume_mesh && tm_patch_volume_debug.getState()) {
			showPVVolumeMesh();
		}
		if (params.pv_show_volume_normals && tls_patch_volume_normals.getState()) {
			showPVVolumeNormals();
		}
		if (params.pv_show_graph_edges && tls_graph_edges.getState()) {
			showPVGraphEdges(object_pose);
		}
		cout << "[TIMING] updatePatchVolumeVisualizer: " << sw.getTime() << " ms." << endl;
	}
}

void ObjectModeler::setVolumeToDebugSphere()
{
	if (params.use_patch_volumes) {
		cout << "setVolumeToDebugSphere not compatible with use_patch_volumes" << endl;
		return;
	}

	opencl_tsdf_ptr->setVolumeToSphere(params.volume_debug_sphere_radius);
}

