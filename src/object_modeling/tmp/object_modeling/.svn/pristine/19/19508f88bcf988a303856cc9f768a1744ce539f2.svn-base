#include "stdafx.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "typedefs.h"
#include "parameters.h"
#include "ObjectModeler.h"

#define DISABLE_OPENNI2_INPUT 1
#ifndef DISABLE_OPENNI2_INPUT
#include "OpenNIInput.h"
#endif

//#define DISABLE_PCL_GRABBER 1
#ifndef DISABLE_PCL_GRABBER
#include "PCLGrabberInput.h"
#endif

// PCL
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

using namespace std;

bool is_not_pcd(fs::path filename)
{
	return filename.extension() != ".pcd";
}

bool is_not_xyz(fs::path filename)
{
	return filename.extension() != ".xyz";
}

bool is_not_rgbd_bare_png(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	// hackish...name can't have "depth" or "mask" in the name
	if (filename.string().find("depth") != string::npos) return true;
	if (filename.string().find("mask") != string::npos) return true;
	return false; // IS a bare rgbd png
}

bool is_not_oni_png_color_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	if (filename.string().find("depth") != string::npos) return true;
	return false; // is an oni png color image file
}

bool is_not_raw_color(fs::path filename)
{
	if (filename.extension() != ".raw") return true;
	if (filename.string().find("depth") != string::npos) return true;
	if (filename.string().find("texture") != string::npos) return true;
	return false; // is a raw file containing "color"
}

bool is_not_evan_color_png(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	if (filename.string().find("dm") != string::npos) return true;
	return false;
}

vector<fs::path> naturalSortRGBD(vector<fs::path> const& input, int which_view)
{
	// ASSUME all file stems end with a number, preceeded by '_'
	map<int, fs::path> path_map;
	for (vector<fs::path>::const_iterator iter = input.begin(); iter != input.end(); ++iter) {
		// get the number
		std::string to_search = iter->stem().string();
		size_t last_number_pos = to_search.find_last_of('_');
		int n = boost::lexical_cast<int>(to_search.substr(last_number_pos+1));
		// also need to make sure it's the correct view
		to_search = to_search.substr(0, last_number_pos);
		last_number_pos = to_search.find_last_of('_');
		int view = boost::lexical_cast<int>(to_search.substr(last_number_pos+1));
		if (view != which_view) continue;
		path_map.insert(make_pair(n, *iter));
	}
	vector<fs::path> result;
	for (map<int, fs::path>::iterator iter = path_map.begin(); iter != path_map.end(); ++iter) {
		result.push_back(iter->second);

		// debug remove:
		cout << "file: " << iter->second << endl;
	}
	return result;
}

bool loadXYXFile(Parameters& params, fs::path filename, CloudT& cloud)
{
	std::ifstream ifs(filename.string().c_str());
	for (int i = 0; i < params.camera_size_x * params.camera_size_y; ++i) {
		if (!ifs.good()) {
			cout << "!ifs.good()" << endl;
			return false;
		}
		std::string line;
		getline(ifs, line);
		std::stringstream ss(line);
		PointT p;
		ss >> p.x >> p.y >> p.z;
		// rgb or bgr??
		int r, g, b;
		ss >> r >> g >> b;
		p.r = r;
		p.g = g;
		p.b = b;
		p.a = 255; // shouldn't matter?
		
		if (p.z == 0) {
			p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
		}
		else {
			// I believe the xyz clouds are in mm:
			p.x /= 1000;
			p.y /= 1000;
			p.z /= 1000;
			// I also believe they flip the y axis
			p.y *= -1;
		}

		cloud.push_back(p);
	}
	if (cloud.size() != params.camera_size_x * params.camera_size_y) {
		cout << "Wrong size: " << cloud.size() << endl;
		return false;
	}
	cloud.width = params.camera_size_x;
	cloud.height = params.camera_size_y;
	cloud.is_dense = false;
	return true;
}

bool loadRGBDTurntableFiles(Parameters& params, fs::path filename, FrameT& frame)
{
	fs::path expected_depth_file = filename.parent_path() / (filename.stem().string() + "_depth.png");
	fs::path expected_mask_file = filename.parent_path() / (filename.stem().string() + "_mask.png");
	bool have_mask = false;

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find: " << expected_depth_file << endl;
		return false;
	}
	if (fs::exists(expected_mask_file)) {
		cout << "found mask file: " << expected_mask_file << endl;
		have_mask = true;
	}

	cv::Mat image_color = cv::imread(filename.string());
	cv::Mat image_depth = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat image_mask;
	if (have_mask) {
		image_mask = cv::imread(expected_mask_file.string(), CV_LOAD_IMAGE_GRAYSCALE);
	}

	// could check all sorts of dimensions here...
	int rows = image_color.rows;
	int cols = image_color.cols;

	// need to project depth into cloud....
	frame.cloud_ptr.reset(new CloudT);
	frame.cloud_ptr->resize(rows * cols);
	frame.cloud_ptr->width = cols;
	frame.cloud_ptr->height = rows;
	frame.cloud_ptr->is_dense = false;

	for (int row = 0; row < image_color.rows; ++row) {
		for (int col = 0; col < image_color.cols; ++col) {
			PointT& p = frame.cloud_ptr->at(col,row);
			p.z = (float)image_depth.at<uint16_t>(row,col) / 1000;
			if (p.z <= 0) {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				p.x = (col - params.camera_center_x) * p.z / params.camera_focal_x;
				p.y = (row - params.camera_center_y) * p.z / params.camera_focal_y;
			}
			p.b = image_color.at<cv::Vec3b>(row,col)[0];
			p.g = image_color.at<cv::Vec3b>(row,col)[1];
			p.r = image_color.at<cv::Vec3b>(row,col)[2];
			p.a = 255;
		}
	}

	if (have_mask) {
		frame.object_mask = image_mask;
	}

	return true;
}

bool loadONIPNGFiles(Parameters& params, fs::path filename, FrameT& frame)
{
	// assume #####-depth.png
	std::string expected_depth_filename = filename.stem().string().substr(0,6) + "depth.png";
	fs::path expected_depth_file = filename.parent_path() / (expected_depth_filename);

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find: " << expected_depth_file << endl;
		return false;
	}

	cv::Mat image_color = cv::imread(filename.string());
	cv::Mat image_depth = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);

	// could check all sorts of dimensions here...
	int rows = image_color.rows;
	int cols = image_color.cols;

	// need to project depth into cloud....
	frame.cloud_ptr.reset(new CloudT);
	frame.cloud_ptr->resize(rows * cols);
	frame.cloud_ptr->width = cols;
	frame.cloud_ptr->height = rows;
	frame.cloud_ptr->is_dense = false;

	for (int row = 0; row < image_color.rows; ++row) {
		for (int col = 0; col < image_color.cols; ++col) {
			PointT& p = frame.cloud_ptr->at(col,row);
			p.z = (float)image_depth.at<uint16_t>(row,col) / 10000; // note this is assuming units of 100um, not 1mm
			if (p.z <= 0) {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				p.x = (col - params.camera_center_x) * p.z / params.camera_focal_x;
				p.y = (row - params.camera_center_y) * p.z / params.camera_focal_y;
			}
			p.b = image_color.at<cv::Vec3b>(row,col)[0];
			p.g = image_color.at<cv::Vec3b>(row,col)[1];
			p.r = image_color.at<cv::Vec3b>(row,col)[2];
			p.a = 255;
		}
	}

	return true;
}

bool loadRAWFiles(Parameters& params, fs::path filename, FrameT& frame)
{
	/*
	Thanks. There are three different image files per time frame. The texture image has (u,v) values (normalized between 0-1) that map from the depth image into the color image.
Depth:
XYZW
Color:
RGBA
Texture:
UV

  I_d=I_c(u*320,v*240) 
	*/

	// assume filename is color####.raw
	// assume depth####.raw
	// assume texture####.raw
	std::string expected_depth_filename = "depth" + filename.stem().string().substr(5,4) + ".raw";
	fs::path expected_depth_file = filename.parent_path() / (expected_depth_filename);
	std::string expected_texture_filename = "texture" + filename.stem().string().substr(5,4) + ".raw";
	fs::path expected_texture_file = filename.parent_path() / (expected_texture_filename);

	if (params.camera_size_x != 320 || params.camera_size_y != 240) {
		cout << "expecting camera_size 320x240" << endl;
		return false;
	}

	ifstream color_stream(filename.string().c_str(), ios::in | ios::binary);
	ifstream depth_stream(expected_depth_file.string().c_str(), ios::in | ios::binary);
	ifstream texture_stream(expected_texture_file.string().c_str(), ios::in | ios::binary);
	if (!color_stream.good()) return false;
	if (!depth_stream.good()) return false;
	if (!texture_stream.good()) return false;

	int cols = params.camera_size_x;
	int rows = params.camera_size_y;

	frame.cloud_ptr.reset(new CloudT);
	frame.cloud_ptr->resize(rows * cols);
	frame.cloud_ptr->width = cols;
	frame.cloud_ptr->height = rows;
	frame.cloud_ptr->is_dense = false;

	const float depth_scale = 1./1000;
	const float color_scale = 255;
	const bool invert_y = true;

#if 0
	// should just be able to load color cv mat image
	cv::Mat color_image_float(rows, cols, CV_32FC4);
	if (!color_stream.good()) return false;
	color_stream.read((char*)color_image_float.data, rows * cols * 4 * sizeof(float));
	cv::imshow("color_image_float", color_image_float);
	// convert to char color
	cv::Mat color_image_u8c4;
	color_image_float.convertTo(color_image_u8c4, CV_8U, color_scale); // rgba
	cv::imshow("color_image_u8c4", color_image_u8c4);
	cout << "debug waitkey in raw load" << endl;
	cv::waitKey();
#endif

	std::vector<cv::Mat> color_images;

	color_images.push_back(cv::Mat(rows, cols, CV_32F));
	color_stream.read((char*)color_images.back().data, rows * cols * sizeof(float));
	color_images.push_back(cv::Mat(rows, cols, CV_32F));
	color_stream.read((char*)color_images.back().data, rows * cols * sizeof(float));
	color_images.push_back(cv::Mat(rows, cols, CV_32F));
	color_stream.read((char*)color_images.back().data, rows * cols * sizeof(float));
	cv::Mat merged;
	cv::merge(color_images, merged);
	cv::cvtColor(merged, merged, CV_RGB2BGR);

	// make sure this is a bgr u8 result
	cv::Mat color_image;
	merged.convertTo(color_image, CV_8U, color_scale);

#if 0
	cv::imshow("merged", merged);
	cout << "debug waitkey in raw load" << endl;
	cv::waitKey();
#endif

	// also just read in "texture" image
#if 0
	cv::Mat texture_image_float(rows, cols, CV_32FC2);
	texture_stream.read((char*)texture_image_float.data, rows * cols * 2 * sizeof(float));
#endif
	cv::Mat texture_image_u(rows, cols, CV_32F);
	cv::Mat texture_image_v(rows, cols, CV_32F);
	texture_stream.read((char*)texture_image_u.data, rows * cols * sizeof(float));
	texture_stream.read((char*)texture_image_v.data, rows * cols * sizeof(float));

	// read in the depth as images as well (channels are not interleved)
	std::vector<cv::Mat> depth_images;
	for (int i = 0; i < 3; ++i) {
		depth_images.push_back(cv::Mat(rows, cols, CV_32FC1));
		depth_stream.read((char*)depth_images.back().data, rows * cols * sizeof(float));
	}
	cv::Mat depth_image;
	cv::merge(depth_images, depth_image);

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			PointT& p = frame.cloud_ptr->at(col,row);
			cv::Vec3f coords = depth_image.at<cv::Vec3f>(row,col);

			if (coords[2] > 0) {
				p.x = coords[0] * depth_scale;
				p.y = coords[1] * depth_scale;
				p.z = coords[2] * depth_scale;
				if (invert_y) {
					p.y *= -1;
				}
			}
			else {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}

			// assign color based on texture lookup
			float u = texture_image_u.at<float>(row, col);
			float v = texture_image_v.at<float>(row, col);
			int color_row = v*rows;
			int color_col = u*cols;
			if (color_row >= 0 && color_row < rows && color_col >= 0 && color_col < cols) {
				cv::Vec3b color = color_image.at<cv::Vec3b>(color_row, color_col);
				p.b = color[0];
				p.g = color[1];
				p.r = color[2];
			}
			else {
				p.r = p.g = p.b = 0;
			}
		}
	}

	return true;
}

bool loadEvanFrame(Parameters& params, fs::path filename, FrameT& frame)
{
	// assume frame######.dm.png
	//std::string expected_depth_filename = filename.stem().string().substr(0,6) + "depth.png";
	std::string expected_depth_filename = filename.stem().string().substr(0,11) + ".dm.png";
	fs::path expected_depth_file = filename.parent_path() / (expected_depth_filename);

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find: " << expected_depth_file << endl;
		return false;
	}

	cv::Mat image_color = cv::imread(filename.string());
	cv::Mat image_depth = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);

	// could check all sorts of dimensions here...
	int rows = image_color.rows;
	int cols = image_color.cols;

	// need to project depth into cloud....
	frame.cloud_ptr.reset(new CloudT);
	frame.cloud_ptr->resize(rows * cols);
	frame.cloud_ptr->width = cols;
	frame.cloud_ptr->height = rows;
	frame.cloud_ptr->is_dense = false;

	float scale = 1000;

	for (int row = 0; row < image_color.rows; ++row) {
		for (int col = 0; col < image_color.cols; ++col) {
			PointT& p = frame.cloud_ptr->at(col,row);
			p.z = (float)image_depth.at<uint16_t>(row,col) / scale;
			if (p.z <= 0) {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				p.x = (col - params.camera_center_x) * p.z / params.camera_focal_x;
				p.y = (row - params.camera_center_y) * p.z / params.camera_focal_y;
			}
			p.b = image_color.at<cv::Vec3b>(row,col)[0];
			p.g = image_color.at<cv::Vec3b>(row,col)[1];
			p.r = image_color.at<cv::Vec3b>(row,col)[2];
			p.a = 255;
		}
	}

	return true;
}

bool loadFreiburgFrame(Parameters& params, fs::path filename_rgb, fs::path filename_depth, FrameT& frame)
{
	if (!fs::exists(filename_rgb)) {
		cout << "couldn't find: " << filename_rgb << endl;
		return false;
	}
	if (!fs::exists(filename_depth)) {
		cout << "couldn't find: " << filename_depth << endl;
		return false;
	}

	cv::Mat image_color = cv::imread(filename_rgb.string());
	cv::Mat image_depth = cv::imread(filename_depth.string(), CV_LOAD_IMAGE_ANYDEPTH);

	// could check all sorts of dimensions here...
	int rows = image_color.rows;
	int cols = image_color.cols;

	// need to project depth into cloud....
	frame.cloud_ptr.reset(new CloudT);
	frame.cloud_ptr->resize(rows * cols);
	frame.cloud_ptr->width = cols;
	frame.cloud_ptr->height = rows;
	frame.cloud_ptr->is_dense = false;

	float factor = 5000;
	for (int row = 0; row < image_color.rows; ++row) {
		for (int col = 0; col < image_color.cols; ++col) {
			PointT& p = frame.cloud_ptr->at(col,row);
			p.z = (float)image_depth.at<uint16_t>(row,col) / factor;
			if (p.z <= 0) {
				p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				p.x = (col - params.camera_center_x) * p.z / params.camera_focal_x;
				p.y = (row - params.camera_center_y) * p.z / params.camera_focal_y;
			}
			p.b = image_color.at<cv::Vec3b>(row,col)[0];
			p.g = image_color.at<cv::Vec3b>(row,col)[1];
			p.r = image_color.at<cv::Vec3b>(row,col)[2];
			p.a = 255;
		}
	}

	return true;
}

bool loadFreiburgAssociateFile(fs::path associate_file, std::vector<std::pair<fs::path, fs::path> > & v)
{
	if (!fs::exists(associate_file) || !fs::is_regular_file(associate_file)) {
		cout << "couldn't find or is not regular file: " << associate_file << endl;
		return false;
	}
	fs::path folder = associate_file.parent_path();

	std::ifstream ifs(associate_file.string().c_str());
	while (ifs.good()) {
		//std::string line;
		//std::getline(ifs, line);
		//		if (!line.empty()) {
		float stamp1, stamp2;
		fs::path path1, path2;
		ifs >> stamp1 >> path1 >> stamp2 >> path2;
		v.push_back(std::make_pair(folder / path1, folder / path2));
	}
	return true;
}

bool setParametersFromCommandLine(int argc, char* argv[], Parameters& params)
{
	// can set both focal lengths to the same thing
	float focal_both = 0;

	// can set all 3 
	int volume_cell_count = 0;

	// set this string, converted to enum
	std::string combined_image_error_string;

	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
		("script", po::value<bool>(&params.script_mode)->zero_tokens(), "script_mode")
		("pause", po::value<bool>(&params.folder_debug_pause_after_every_frame)->zero_tokens(), "folder_debug_pause_after_every_frame: pause after every frame (cv::waitKey())")
		("pause-fail", po::value<bool>(&params.folder_debug_pause_after_failure)->zero_tokens(), "folder_debug_pause_after_failure: pause after frames returning false (cv::waitKey())")
		("repeat", po::value<bool>(&params.folder_debug_keep_same_frame)->zero_tokens(), "folder_debug_keep_same_frame: repeat the first frame over and over and over (debugging)")

		("openni", po::value<bool>(&params.live_input_openni)->zero_tokens(), "live_input_openni")

		("disable-viewer", po::value<bool>(&params.disable_cloud_viewer)->zero_tokens(), "disable_cloud_viewer")
		("disable-cout-buffer", po::value<bool>(&params.disable_cout_buffer)->zero_tokens(), "disable_cout_buffer")

		("input", po::value<fs::path>(&params.folder_input), "folder_input: input folder with pcd files.  Setting this disables live input")
		("load-object-poses", po::value<bool>(&params.load_object_poses)->zero_tokens(), "load_object_poses")
		("debug-do-load-pv-info", po::value<bool>(&params.debug_do_load_pv_info)->zero_tokens(), "debug_do_load_pv_info")
		("input-file-object-poses", po::value<fs::path>(&params.file_input_object_poses), "file_input_object_poses")
		("input-file-loop-poses", po::value<fs::path>(&params.file_input_loop_poses), "file_input_loop_poses")
		("input-file-loop-which-pvs", po::value<fs::path>(&params.file_input_loop_which_pvs), "file_input_loop_which_pvs")
		("fs", po::value<int>(&params.folder_frame_start), "folder_frame_start")
		("fi", po::value<int>(&params.folder_frame_increment), "folder_frame_increment")
		("fe", po::value<int>(&params.folder_frame_end), "folder_frame_end (exclusive) (<= 0 disables)")
		("fxyz", po::value<bool>(&params.folder_is_xyz_files)->zero_tokens(), "folder_is_xyz_files")
		("frgbd", po::value<bool>(&params.folder_is_rgbd_turntable_files)->zero_tokens(), "folder_is_rgbd_turntable_files")
		("fonipng", po::value<bool>(&params.folder_is_oni_png_files)->zero_tokens(), "folder_is_oni_png_files")
		("fraw", po::value<bool>(&params.folder_is_raw_files)->zero_tokens(), "folder_is_raw_files")
		("ffreiburg", po::value<bool>(&params.folder_is_freiburg_files)->zero_tokens(), "folder_is_freiburg_files")
		("fevan", po::value<bool>(&params.folder_is_evan_files)->zero_tokens(), "folder_is_evan_files")

		("max", po::value<float>(&params.max_depth_in_input_cloud), "max_depth_in_input_cloud: filter out all points beyond this depth in input cloud before sending to object modeler")
		("output", po::value<fs::path>(&params.folder_output), "folder_output: where all written files go.  Defaults to 'dump'.")
		("save-input", po::value<bool>(&params.save_input)->zero_tokens(), "save_input")
		("save-input-only", po::value<bool>(&params.save_input_only)->zero_tokens(), "save_input_only: This puts the program in frame grabber mode.")
		("save-only-max-fps", po::value<float>(&params.save_input_only_max_fps), "save_input_only_max_fps: Sets the max FPS recorded when doing save_input_only")
		("save-input-images", po::value<bool>(&params.save_input_images)->zero_tokens(), "save_input_images")
		("save-tables", po::value<bool>(&params.save_tables_of_values)->zero_tokens(), "save_tables_of_values")
		("save-poses", po::value<bool>(&params.save_poses)->zero_tokens(), "save_poses")
		("save-objects-pcd", po::value<bool>(&params.save_objects_pcd)->zero_tokens(), "save_objects_pcd")
		("save-objects-png", po::value<bool>(&params.save_objects_png)->zero_tokens(), "save_objects_png")
		("save-cdi", po::value<bool>(&params.save_cdi_images)->zero_tokens(), "save_cdi_images")
		("save-viewer", po::value<bool>(&params.save_viewer_screenshots)->zero_tokens(), "save_viewer_screenshots")
		("save-render-align", po::value<bool>(&params.save_render_for_alignment)->zero_tokens(), "save_render_for_alignment")
		("save-seg-di", po::value<bool>(&params.save_segment_debug_images)->zero_tokens(), "save_segment_debug_images")

		("render-after", po::value<bool>(&params.render_after)->zero_tokens(), "render_after")
		("save-render-after", po::value<bool>(&params.save_render_after)->zero_tokens(), "save_render_after")

		("calibrate", po::value<bool>(&params.folder_do_calibrate)->zero_tokens(), "folder_do_calibrate: run calibration instead of anything else.  See the code...this is a one-off hack that Peter used.")
		("rvs", po::value<float>(&params.render_viewer_scale), "render_viewer_scale: Scale for renderers triggered by keypresses to the PCL viewer")
		("sif", po::value<bool>(&params.set_inactive_on_failure)->zero_tokens(), "set_inactive_on_failure")
		("show-axes", po::value<bool>(&params.show_axes)->zero_tokens(), "show_axes: The axes are at the camera location.")
		("white-bg", po::value<bool>(&params.white_background)->zero_tokens(), "white_background")

		("cam-sx", po::value<int>(&params.camera_size_x), "camera_size_x")
		("cam-sy", po::value<int>(&params.camera_size_y), "camera_size_y")
		("cam-f", po::value<float>(&focal_both), "focal_length: focal length for both x and y.  Sets params.camera_focal_x = params.camera_focal_y = focal_length;")
		("cam-cx", po::value<float>(&params.camera_center_x), "camera_center_x")
		("cam-cy", po::value<float>(&params.camera_center_y), "camera_center_y")
		("cam-fx", po::value<float>(&params.camera_focal_x), "camera_focal_x")
		("cam-fy", po::value<float>(&params.camera_focal_y), "camera_focal_y")
		("cam-fix", po::value<bool>(&params.correct_input_camera_params)->zero_tokens(), "correct_input_camera_params")
		("cam-zmin", po::value<float>(&params.camera_z_min), "camera_z_min")
		("cam-zmax", po::value<float>(&params.camera_z_max), "camera_z_max")

		("fdi", po::value<bool>(&params.features_debug_images)->zero_tokens(), "features_debug_images")
		("fff", po::value<bool>(&params.features_frame_to_frame)->zero_tokens(), "features_frame_to_frame")

		("frp", po::value<float>(&params.ransac_probability), "ransac_probability")
		("frv", po::value<int>(&params.ransac_verbosity), "ransac_verbosity")
		("frmi", po::value<int>(&params.ransac_max_iterations), "ransac_max_iterations")
		("fri", po::value<int>(&params.ransac_min_inliers), "ransac_min_inliers")

		("oclpath", po::value<fs::path>(&params.opencl_cl_path), "opencl_cl_path")
		("ocln", po::value<bool>(&params.opencl_nvidia)->zero_tokens(), "opencl_nvidia: use nvidia opencl (instead of Intel default)")
		("oclg", po::value<bool>(&params.opencl_context_gpu)->zero_tokens(), "opencl_context_gpu: force GPU")
		("oclc", po::value<bool>(&params.opencl_context_cpu)->zero_tokens(), "opencl_context_cpu: force CPU")
		("ocld", po::value<bool>(&params.opencl_debug)->zero_tokens(), "opencl_debug: compile all opencl code with debug (Intel only)")
		("ocl-fast", po::value<bool>(&params.opencl_fast_math)->zero_tokens(), "opencl_fast_math")

		("of", po::value<bool>(&params.use_features)->zero_tokens(), "use_features: use ransac-based feature matching to initialize combined optimization")
		("oc", po::value<bool>(&params.use_combined_optimization)->zero_tokens(), "use_combined_optimization: use the combined ICP and Color OpenCL optimization (the core algorithm)")

		("icox", po::value<float>(&params.initial_centroid_offset_x), "initial_centroid_offset_x")
		("icoy", po::value<float>(&params.initial_centroid_offset_y), "initial_centroid_offset_y")
		("icoz", po::value<float>(&params.initial_centroid_offset_z), "initial_centroid_offset_z")
		("icic", po::value<bool>(&params.initial_centroid_image_center)->zero_tokens(), "initial_centroid_image_center: Instead of using the centroid of first-frame object points, center in x- and y- and use only the z mean")
		("icfix", po::value<bool>(&params.initial_centroid_fixed)->zero_tokens(), "initial_centroid_fixed: Start the volume at (icox, icoy, icoz) regardless of object points")

		// object mask
		("mo", po::value<bool>(&params.mask_object)->zero_tokens(), "mask_object: object mode")
		("mh", po::value<bool>(&params.mask_hand)->zero_tokens(), "mask_hand in object mode")
		("minput", po::value<bool>(&params.mask_input_if_present)->zero_tokens(), "mask_input_if_present")
		("mdi", po::value<bool>(&params.mask_debug_images)->zero_tokens(), "mask_debug_images")
		("moms", po::value<int>(&params.mask_object_min_size), "mask_object_min_size")
		("micf", po::value<float>(&params.mask_initial_search_contraction_factor), "mask_initial_search_contraction_factor")
		("mcf", po::value<float>(&params.mask_search_contraction_factor), "mask_search_contraction_factor")
		("mcdd", po::value<float>(&params.mask_connected_max_depth_difference), "mask_connected_max_depth_difference")
		("mddd", po::value<float>(&params.mask_disconnected_max_depth_difference), "mask_disconnected_max_depth_difference")
		("mgdd", po::value<float>(&params.mask_global_max_depth_difference), "mask_global_max_depth_difference")
		("moe", po::value<int>(&params.mask_object_erode), "mask_object_erode")
		("moofs", po::value<bool>(&params.mask_object_use_only_first_segment)->zero_tokens(), "mask_object_use_only_first_segment")
		("mar", po::value<bool>(&params.mask_always_reset_search_region)->zero_tokens(), "mask_always_reset_search_region")
		("mdec", po::value<bool>(&params.mask_debug_every_component)->zero_tokens(), "mask_debug_every_component")
		("mff", po::value<bool>(&params.mask_floodfill)->zero_tokens(), "mask_floodfill: expand the MASKED OUT pixels with a floodfill based on color")
		("mffd", po::value<float>(&params.mask_floodfill_expand_diff), "mask_floodfill_expand_diff: param for mask_floodfill ")

		// hand mask
		("mhl", po::value<bool>(&params.mask_hand_hist_learn)->zero_tokens(), "mask_hand_hist_learn")
		("mol", po::value<bool>(&params.mask_object_hist_learn)->zero_tokens(), "mask_object_hist_learn")
		("mhlhb", po::value<int>(&params.mask_hand_learn_hbins), "mask_hand_learn_hbins")
		("mhlsb", po::value<int>(&params.mask_hand_learn_sbins), "mask_hand_learn_sbins")
		("mhlvb", po::value<int>(&params.mask_hand_learn_vbins), "mask_hand_learn_vbins")
		("mhbt", po::value<float>(&params.mask_hand_backproject_thresh), "mask_hand_backproject_thresh")
		("mhhe", po::value<int>(&params.mask_hand_erode), "mask_hand_erode")
		("mhhd", po::value<int>(&params.mask_hand_dilate), "mask_hand_dilate")
		("mhmcb", po::value<int>(&params.mask_hand_min_component_area_before_morphology), "mask_hand_min_component_area_before_morphology")
		("mhmca", po::value<int>(&params.mask_hand_min_component_area_after_morphology), "mask_hand_min_component_area_after_morphology")
		("mh-hand-file", po::value<fs::path>(&params.mask_hand_hist_filename), "mask_hand_hist_filename")
		("mh-object-file", po::value<fs::path>(&params.mask_object_hist_filename), "mask_object_hist_filename")

		("vn", po::value<int>(&volume_cell_count), "volume_cell_count (per cube edge)")
		("vnx", po::value<int>(&params.volume_cell_count_x), "volume_cell_count_x")
		("vny", po::value<int>(&params.volume_cell_count_y), "volume_cell_count_y")
		("vnz", po::value<int>(&params.volume_cell_count_z), "volume_cell_count_z")
		("vs", po::value<float>(&params.volume_cell_size), "volume_cell_size (m)")
		("vmw", po::value<float>(&params.volume_max_weight), "volume_max_weight")
		("vdsm", po::value<bool>(&params.volume_debug_show_max_points)->zero_tokens(), "volume_debug_show_max_points")
		("v-sphere", po::value<bool>(&params.volume_debug_sphere)->zero_tokens(), "volume_debug_sphere")
		("v-sphere-r", po::value<float>(&params.volume_debug_sphere_radius), "volume_debug_sphere_radius")


		("wp", po::value<float>(&params.combined_weight_icp_points), "combined_weight_icp_points")
		("wc", po::value<float>(&params.combined_weight_color), "combined_weight_color")
		("bs", po::value<int>(&params.color_blur_size), "color_blur_size: blur kernel to apply to frame (target) pixels for color error.  Recommend 5 (default) to provide gradient for optimization")
		("bap", po::value<bool>(&params.color_blur_after_pyramid)->zero_tokens(), "color_blur_after_pyramid: apply the color_blur_size to each octave instead of just to the starting image ")
		("cp", po::value<bool>(&params.combined_pause_every_eval)->zero_tokens(), "combined_pause_every_eval: debugging, pause combined error function after every evaluation")
		("c-verbose", po::value<bool>(&params.combined_verbose)->zero_tokens(), "combined_verbose")
		("cdi", po::value<bool>(&params.combined_debug_images)->zero_tokens(), "combined_debug_images")
		("cdis", po::value<float>(&params.combined_debug_images_scale), "combined_debug_images_scale")
		("cgn", po::value<bool>(&params.combined_gauss_newton)->zero_tokens(), "combined_gauss_newton: briefly tried gauss-newton instead of levenburg-mardquadt.  See code.")
		("cgn-gpu-full", po::value<bool>(&params.combined_gauss_newton_gpu_full)->zero_tokens(), "combined_gauss_newton_gpu_full: even newer GPU gauss newton")
		("cgn-max-iter", po::value<int>(&params.combined_gauss_newton_max_iterations), "combined_gauss_newton_max_iterations: param for combined_gauss_newton")
		("cgn-min-delta", po::value<float>(&params.combined_gauss_newton_min_delta_to_continue), "combined_gauss_newton_min_delta_to_continue")
		("crs", po::value<float>(&params.combined_render_scale), "combined_render_scale: set to 1./2 or 1./4 to speed up optimization at the cost of some accuracy")
		("co", po::value<int>(&params.combined_octaves), "combined_octaves: multiscale optimization")
		("cie", po::value<std::string>(&combined_image_error_string), "combined_image_error_string")
		("c-pause-all", po::value<bool>(&params.combined_debug_pause_after_icp_all)->zero_tokens(), "associat")
		("c-pause-each", po::value<bool>(&params.combined_debug_pause_after_icp_iteration)->zero_tokens(), "combined_debug_pause_after_icp_iteration")
		("c-debug-normal", po::value<bool>(&params.combined_debug_normal_eq)->zero_tokens(), "combined_debug_normal_eq")
		("c-debug-single", po::value<bool>(&params.combined_debug_single_kernel)->zero_tokens(), "combined_debug_single_kernel")
		("c-stats", po::value<bool>(&params.combined_compute_error_statistics)->zero_tokens(), "combined_compute_error_statistics: these take ~5ms per full frame error on my home machine")
		("c-min-points", po::value<int>(&params.combined_min_rendered_point_count), "combined_min_rendered_point_count")
		("c-show-render", po::value<bool>(&params.combined_show_render)->zero_tokens(), "combined_show_render")

		("emt", po::value<float>(&params.error_max_t), "error_max_t")
		("emr", po::value<float>(&params.error_max_r), "error_max_r")
		("ec", po::value<float>(&params.error_change), "error_change")
		("emii", po::value<float>(&params.error_min_inlier_fraction_icp), "error_min_inlier_fraction_icp")
		("emic", po::value<float>(&params.error_min_inlier_fraction_color), "error_min_inlier_fraction_color")
		("emrp", po::value<int>(&params.error_min_rendered_points), "error_min_rendered_points")
		("emf", po::value<int>(&params.error_min_output_frames), "error_min_output_frames")
		("emev", po::value<float>(&params.error_min_eigenvalue), "error_min_eigenvalue")
		("ert", po::value<float>(&params.error_rank_threshold), "error_rank_threshold")
		("eur", po::value<bool>(&params.error_use_rank)->zero_tokens(), "error_use_rank")

		("id", po::value<float>(&params.icp_max_distance), "icp_max_distance: max distance to matching point before discarding as outlier (meters)")
		("ina", po::value<float>(&params.icp_normals_angle), "icp_normals_angle: max angle between corresponding ICP points (degrees)")
		("imt", po::value<float>(&params.icp_min_translation_to_continue), "icp_min_translation_to_continue: if multiple ICP iterations, continue if translation magnitude greater than this")
		("imr", po::value<float>(&params.icp_min_rotation_to_continue), "icp_min_rotation_to_continue: if multipel ICP iterations, continue if rotation greater than this (degrees)")
		("imi", po::value<int>(&params.icp_max_iterations), "icp_max_iterations.  Defaults to 1 as combined optimization takes place within a single iteration")

		("n-smooth", po::value<int>(&params.normals_smooth_iterations), "normals_smooth_iterations")
		("n-opencl-debug", po::value<bool>(&params.normals_opencl_debug)->zero_tokens(), "normals_opencl_debug")
		
		("mesh-show", po::value<bool>(&params.mesh_show)->zero_tokens(), "mesh_show")
		("mesh-weights", po::value<bool>(&params.mesh_marching_cubes_weights)->zero_tokens(), "mesh_marching_cubes_weights")

		("max-sigmas", po::value<float>(&params.max_depth_sigmas), "max_depth_sigmas")

		("seg-a", po::value<float>(&params.segments_max_angle), "segments_max_angle")
		("seg-min-size", po::value<int>(&params.segments_min_size), "segments_min_size")
		("seg-di", po::value<bool>(&params.segment_debug_images)->zero_tokens(), "segment_debug_images")

		("pv", po::value<bool>(&params.use_patch_volumes)->zero_tokens(), "use_patch_volumes")
		("pv-bi", po::value<float>(&params.pv_initial_border_size), "pv_initial_border_size")
		("pv-be", po::value<float>(&params.pv_expand_border_size), "pv_expand_border_size")
		("pv-max-age-loop", po::value<int>(&params.pv_max_age_before_considered_loop), "pv_max_age_before_considered_loop")
		("pv-age-deallocate", po::value<int>(&params.pv_age_to_deallocate), "pv_age_to_deallocate")
		("pv-loop", po::value<bool>(&params.pv_loop_closure)->zero_tokens(), "pv_loop_closure")
		("pv-loop-min-coverage", po::value<float>(&params.pv_loop_min_frame_coverage), "pv_loop_min_frame_coverage")
		("pv-loop-max-angle", po::value<float>(&params.pv_loop_max_normal_angle), "pv_loop_max_normal_angle")
		("pv-min-size", po::value<int>(&params.pv_min_size_to_create_new), "pv_min_size_to_create_new")
		("pv-loop-icp-iter", po::value<int>(&params.pv_loop_icp_max_iterations), "pv_loop_icp_max_iterations")
		("pv-show-edges", po::value<bool>(&params.pv_show_volume_edges)->zero_tokens(), "pv_show_volume_edges")
		("pv-show-mesh", po::value<bool>(&params.pv_show_volume_mesh)->zero_tokens(), "pv_show_volume_mesh")
		("pv-show-normals", po::value<bool>(&params.pv_show_volume_normals)->zero_tokens(), "pv_show_volume_normals")
		("pv-show-graph-edges", po::value<bool>(&params.pv_show_graph_edges)->zero_tokens(), "pv_show_graph_edges")
		("pv-mesh-alpha", po::value<float>(&params.pv_mesh_alpha), "pv_mesh_alpha")
		("pv-optimize-always", po::value<bool>(&params.pv_debug_optimize_always)->zero_tokens(), "pv_debug_optimize_always")
		("pv-graph-iter", po::value<int>(&params.pv_pose_graph_iterations), "pv_pose_graph_iterations")
		("pv-edge-loop-factor", po::value<float>(&params.pv_edge_loop_information_factor), "pv_edge_loop_information_factor")
		("pvr-show", po::value<bool>(&params.pv_debug_show_render_for_alignment)->zero_tokens(), "pv_debug_show_render_for_alignment")
		("pv-loop-di", po::value<bool>(&params.pv_loop_debug_images)->zero_tokens(), "pv_loop_debug_images")
		("pv-verbose", po::value<bool>(&params.pv_verbose)->zero_tokens(), "pv_verbose")
		("pv-max-side", po::value<int>(&params.pv_max_side_voxel_count), "pv_max_side_voxel_count")
		("pv-split-overlap", po::value<int>(&params.pv_split_voxel_overlap), "pv_split_voxel_overlap")
		("pv-loop-skip", po::value<int>(&params.pv_loop_frame_skip), "pv_loop_frame_skip")
		("pv-loop-features", po::value<bool>(&params.pv_loop_features)->zero_tokens(), "pv_loop_features")
		("pv-debug-vis", po::value<bool>(&params.pv_debug_update_visualizer)->zero_tokens(), "pv_debug_update_visualizer")
		("pv-keyframe-debug", po::value<bool>(&params.pv_keyframe_debug)->zero_tokens(), "pv_keyframe_debug")
		("pv-keyframe-create-d", po::value<float>(&params.pv_keyframe_max_distance_create), "pv_keyframe_max_distance_create")
		("pv-keyframe-create-a", po::value<float>(&params.pv_keyframe_max_angle_create), "pv_keyframe_max_angle_create")
		("pv-keyframe-match-d", po::value<float>(&params.pv_keyframe_max_distance_match), "pv_keyframe_max_distance_match")
		("pv-keyframe-match-a", po::value<float>(&params.pv_keyframe_max_angle_match), "pv_keyframe_max_angle_match")
		("pv-debug-add-volumes", po::value<bool>(&params.pv_debug_add_volumes)->zero_tokens(), "pv_debug_add_volumes")
		("pv-max-mb", po::value<int>(&params.pv_max_mb_allocated), "pv_max_mb_allocated")
		("pv-cov", po::value<bool>(&params.pv_use_covariance_to_create)->zero_tokens(), "pv_use_covariance_to_create")
		("pv-print-edges", po::value<bool>(&params.pv_debug_print_edges)->zero_tokens(), "pv_debug_print_edges")

		("pg", po::value<bool>(&params.use_patch_grid)->zero_tokens(), "use_patch_grid")
		("pg-size", po::value<int>(&params.pg_size), "pg_size")
		("pg-border", po::value<int>(&params.pg_border), "pg_border")
		;
	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (std::exception & e) {
		cout << e.what() << endl;
		cout << desc << endl;
		cout << e.what() << endl;
		return false;
	}

	if (vm.count("help")) {
		cout << "desc" << endl;
		return false;
	}

	// override all 3 dimensions
	if (volume_cell_count > 0) {
		params.volume_cell_count_x = params.volume_cell_count_y = params.volume_cell_count_z = volume_cell_count;
	}

	// set camera center based on size
	if (!vm.count("cam-cx")) params.camera_center_x = (params.camera_size_x - 1) * 0.5;
	if (!vm.count("cam-cy")) params.camera_center_y = (params.camera_size_y - 1) * 0.5;

	// override focal lengths
	if (focal_both != 0) {
		params.camera_focal_x = params.camera_focal_y = focal_both;
	}

	// if using folder input, use "hand.yml" and "object.yml" in that folder
	if (!vm.count("mh-hand-file")) {
		if (!params.folder_input.empty()) {
			params.mask_hand_hist_filename = params.folder_input / params.mask_default_hand_hist_filename;
		}
		else {
			params.mask_hand_hist_filename = params.mask_default_hand_hist_filename;
		}
		cout << "Hand histogram filename set to: " << params.mask_hand_hist_filename << endl;
	}
	if (!vm.count("mh-object-file")) {
		if (!params.folder_input.empty()) {
			params.mask_object_hist_filename = params.folder_input / params.mask_default_object_hist_filename;
		}
		else {
			params.mask_object_hist_filename = params.mask_default_object_hist_filename;
		}
		cout << "Object histogram filename set to: " << params.mask_object_hist_filename << endl;
	}

	// set the combined image error type
	if (combined_image_error_string == "") cout << "Using default combined_image_error" << endl;
	else if (combined_image_error_string == "y") params.combined_image_error = Parameters::IMAGE_ERROR_Y;
	else if (combined_image_error_string == "cbcr") params.combined_image_error = Parameters::IMAGE_ERROR_CBCR;
	else if (combined_image_error_string == "ycbcr") params.combined_image_error = Parameters::IMAGE_ERROR_YCBCR;
	else if (combined_image_error_string == "lab") params.combined_image_error = Parameters::IMAGE_ERROR_LAB;
	else if (combined_image_error_string == "none") params.combined_image_error = Parameters::IMAGE_ERROR_NONE;
	else {
		cout << "Unknown image error string: " << combined_image_error_string << endl;
		return false;
	}


	////////////////////////////
	// check values from options
	// empty folder means live input
	if (!params.folder_input.empty() && !fs::exists(params.folder_input)) {
		cerr << "invalid folder / file: " << params.folder_input << endl;
		return false;
	}

	if (params.folder_frame_increment < 1) {
		cerr << "frame_increment must be >= 1" << endl;
		return false;
	}

	if (!params.save_input_only && ! (params.use_combined_optimization || params.use_features) ) {
		cerr << "must specify some alignment method" << endl;
		return false;
	}

	if (params.mask_hand_hist_learn && params.mask_object_hist_learn) {
		cerr << "params.mask_hand_hist_learn && params.mask_object_hist_learn" << endl;
		return false;
	}

	if (params.opencl_context_cpu && params.opencl_context_gpu) {
		cerr << "params.opencl_context_cpu && params.opencl_context_gpu" << endl;
		return false;
	}

	if (!params.folder_input.empty() && params.save_input) {
		cerr << "!params.folder_input.empty() && params.save_input_clouds" << endl;
		return false;
	}

	for (int i = 1; i < argc; i++) {
		params.full_command_line += std::string(argv[i]) + " ";
	}
	cout << "params.full_command_line: " << endl << params.full_command_line << endl;

	return true;
}

void runFromFolder(Parameters& params)
{
	// create the object modeler
	ObjectModeler object_modeler(params);
	object_modeler.setActive(true);

	// get the pcd files to load
	vector<fs::path> files;
	std::map<fs::path, fs::path> associated_file_map;
	if (params.folder_is_xyz_files) {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_xyz);
		sort(files.begin(), files.end());
	}
	else if (params.folder_is_rgbd_turntable_files) {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_rgbd_bare_png);
		// gotta deal with the "unnatural" sorting in turntable files
		files = naturalSortRGBD(files, 1 /* which view */);
	}
	else if (params.folder_is_oni_png_files) {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_oni_png_color_image);
		sort(files.begin(), files.end());
	}
	else if (params.folder_is_raw_files) {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_raw_color);
		sort(files.begin(), files.end());
	}
	else if (params.folder_is_freiburg_files) {
		// read the association, put rgb in files, depth in map
		std::vector<std::pair<fs::path, fs::path> > v;
		loadFreiburgAssociateFile(params.folder_input, v);
		for (size_t i = 0; i < v.size(); ++i) {
			fs::path rgb_file, depth_file;
			if (params.freiburg_associate_depth_first) {
				depth_file = v[i].first;
				rgb_file = v[i].second;
			}
			else {
				rgb_file = v[i].first;
				depth_file = v[i].second;
			}
			files.push_back(rgb_file);
			associated_file_map[rgb_file] = depth_file;
		}
	}
	else if (params.folder_is_evan_files) {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_evan_color_png);
		sort(files.begin(), files.end());
	}
	else {
		remove_copy_if(fs::directory_iterator(params.folder_input), fs::directory_iterator(), back_inserter(files), is_not_pcd);
		sort(files.begin(), files.end());
	}
	

	// debug
	cout << "Found " << files.size() << " files in " << params.folder_input << endl;

	int frame_counter = 0;
	for(vector<fs::path>::iterator file_iter = files.begin(); file_iter != files.end(); ++file_iter, ++frame_counter) {
		fs::path file = *file_iter;
		if (params.folder_debug_keep_same_frame) --file_iter;
		if (frame_counter < params.folder_frame_start) continue;
		if (params.folder_frame_end > 0 && frame_counter >= params.folder_frame_end) continue;
		if ((frame_counter - params.folder_frame_start) % params.folder_frame_increment != 0) continue;

		cout << "loading: " << file << endl;
		bool hold_this_frame = params.folder_debug_pause_after_every_frame;

		// load cloud
		// a frame to fill in from the cloud
		FrameT frame_current;
		frame_current.cloud_ptr.reset(new CloudT);
		if (params.folder_is_xyz_files) {
			if (!loadXYXFile(params, file, *frame_current.cloud_ptr)) {
				cout << "loadXYXFile failed for file: " << file << endl;
				exit(1);
			}
		}
		else if (params.folder_is_rgbd_turntable_files) {
			if (!loadRGBDTurntableFiles(params, file, frame_current)) {
				cout << "loadRGBDTurntableFiles failed: " << file << endl;
				exit(1);
			}
		}
		else if (params.folder_is_oni_png_files) {
			if (!loadONIPNGFiles(params, file, frame_current)) {
				cout << "loadONIPNGFiles failed: " << file << endl;
				exit(1);
			}
		}
		else if (params.folder_is_raw_files) {
			if (!loadRAWFiles(params, file, frame_current)) {
				cout << "loadRAWFiles failed: " << file << endl;
				exit(1);
			}
		}
		else if (params.folder_is_freiburg_files) {
			if (!loadFreiburgFrame(params, file, associated_file_map[file], frame_current)) {
				cout << "loadFreiburgFrame failed: " << file << " -> " << associated_file_map[file] << endl;
				exit(1);
			}
		}
		else if (params.folder_is_evan_files) {
			if (!loadEvanFrame(params, file, frame_current)) {
				cout << "loadEvanFrame failed" << endl;
				exit(1);
			}
		}
		else {
			try {
				if (pcl::io::loadPCDFile (file.string(), *frame_current.cloud_ptr) == -1) {
					cout << "load failed for file: " << file.string() << endl;
					exit(1);
				}
			}
			catch (pcl::InvalidConversionException e) {
				cout << "pcl::InvalidConversionException: " << e.what() << endl;
				cout << "Will try to load as POINTXYZRGB and convert to RGBA" << endl;
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_xyzrgb_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
				if (pcl::io::loadPCDFile (file.string(), *temp_xyzrgb_cloud_ptr) == -1) {
					cout << "load failed for file: " << file.string() << endl;
					exit(1);
				}
				pcl::copyPointCloud(*temp_xyzrgb_cloud_ptr, *frame_current.cloud_ptr);
				// right, that doesn't do color (since field names are different)
				for (size_t i = 0; i < temp_xyzrgb_cloud_ptr->size(); i++) {
					frame_current.cloud_ptr->at(i).r = temp_xyzrgb_cloud_ptr->at(i).r;
					frame_current.cloud_ptr->at(i).g = temp_xyzrgb_cloud_ptr->at(i).g;
					frame_current.cloud_ptr->at(i).b = temp_xyzrgb_cloud_ptr->at(i).b;
				}
			}
		}
		/////////// end load

		bool process_result = object_modeler.processFrame(frame_current);
		cout << "processFrame result: " << process_result << endl;

		if (!process_result && params.folder_debug_pause_after_failure) hold_this_frame = true;

		//////////////////////////
		// hold this frame?
		if (hold_this_frame) {
			int k = -1;
			cout << "waiting for key at end of frame" << endl;
			while (k < 0) {
				object_modeler.showQueuedImages();
				k = cv::waitKey(1);
			}
		}
		else {
			cv::waitKey(1);
		}

		if (object_modeler.wasStopped()) {
			break;
		}
	}

	if (params.script_mode) {
		object_modeler.generateMesh();
		object_modeler.stop();
		if (!params.disable_cloud_viewer) {
			while (!object_modeler.wasStopped()) {
				cout << "Waiting for object_modeler to stop()..." << endl;
				boost::posix_time::seconds sleepTime(1); 
				boost::this_thread::sleep(sleepTime);
			}
		}
	}
	else {
		cout << "continuing to process waitkey after last input file" << endl;
		while (!object_modeler.wasStopped()) {
			object_modeler.showQueuedImages();
			object_modeler.processWaitKey();
		}
	}
}




int main(int argc, char* argv[])
{
	Parameters params;
	if (!setParametersFromCommandLine(argc, argv, params)) exit(1);

	if (!params.disable_cout_buffer) {
		// windows: buffer stdout better!
		const int console_buffer_size = 4096;
		char buf[console_buffer_size];
		setvbuf(stdout, buf, _IOLBF, console_buffer_size);
	}

	if (params.folder_input.empty()) {
		cout << "folder_input.empty()...assuming live input..." << endl;

		if (params.live_input_openni) {
#ifdef DISABLE_OPENNI2_INPUT
			cout << "OpenNIInput temporarily disabled" << endl;
#else
			OpenNIInput openni_input(params);
			openni_input.run();
#endif
		}
		else {
#ifdef DISABLE_PCL_GRABBER
			cout << "PCLGrabberInput temporarily disabled" << endl;
#else
			PCLGrabberInput pcl_grabber_input(params);
			pcl_grabber_input.run();
#endif
		}
	}
	else {
		runFromFolder(params);
	}

	return 0;
}


