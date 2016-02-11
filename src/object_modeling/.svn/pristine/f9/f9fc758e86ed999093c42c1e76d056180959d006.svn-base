#include "frame_provider_png.h"

FrameProviderPNG::FrameProviderPNG(fs::path folder, float depth_factor, bool expect_pose, fs::path camera_list_txt)
	: depth_factor_(depth_factor),
	expect_pose_(expect_pose)
{
	std::remove_copy_if(fs::directory_iterator(folder), fs::directory_iterator(), std::back_inserter(files_), is_not_png_color_image);
	std::sort(files_.begin(), files_.end());
	file_iter_ = files_.begin();
	cout << "Found " << files_.size() << " frames" << endl;

	if (!camera_list_txt.empty()) {
		/*
		5716
1 0 0 0 0.64 1.28 0.46 
1 0 0 0 0.64 1.28 0.46 
-1
1 0 0 0 0.64 1.28 0.46 
1 0 0 0 0.64 1.28 0.46 
-1
1 0 0 0 0.64 1.28 0.46 
1 0 0 0 0.64 1.28 0.46 
-1
		*/
		fs::path filename = folder / camera_list_txt;
		std::fstream file(filename.string().c_str(), std::ios::in);
		std::string s;
		int camera_list_size;
		file >> camera_list_size;
		std::getline(file, s); // eat line
		camera_list_vec_.resize(camera_list_size);
		for (int i = 0; i < camera_list_size; ++i) {
			std::getline(file, s);
			camera_list_vec_[i] = EigenUtilities::stringToTransform(s);
			// eat next 2 lines
			std::getline(file, s);
			std::getline(file, s);
		}
	}
}

bool FrameProviderPNG::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter_ >= files_.end()) return false;

	// get file and increment
	fs::path filename = *file_iter_++;

	// assume #####-depth.png
	std::string frame_index_string = filename.stem().string().substr(0,6);
	std::string expected_depth_filename = frame_index_string + "depth.png";
	fs::path expected_depth_file = filename.parent_path() / expected_depth_filename;

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find expected_depth_file: " << expected_depth_file << endl;

		// try a second format
		// frame########.png
		// this is getting silly
		frame_index_string = filename.stem().string().substr(5,9);
		expected_depth_filename = "depth"+frame_index_string+".png";
		expected_depth_file = filename.parent_path() / expected_depth_filename;
		if (!fs::exists(expected_depth_file)) {
			cout << "couldn't find expected_depth_file : " << expected_depth_file << endl;
			return false;
		}
	}


	cv::Mat color_bgr = cv::imread(filename.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	cv::Mat image_depth_png = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);
	image_depth_png.convertTo(depth, CV_32F, 1./depth_factor_);

	if (expect_pose_) {
		// assume #####-framePose.txt
		std::string expected_pose_filename = filename.stem().string().substr(0,6) + "framePose.txt";
		fs::path expected_pose_file = filename.parent_path() / expected_pose_filename;
		std::ifstream ifs (expected_pose_file.c_str());
		if (!ifs.good()) {
			cout << "problems with: " << expected_pose_filename << endl;
			return false;
		}
		std::string line;
		std::getline(ifs, line);
		last_frame_pose_ = EigenUtilities::stringToTransform(line);
	}
	else if (!camera_list_vec_.empty()) {
		int frame_index = atoi(frame_index_string.c_str());
		last_frame_pose_ = camera_list_vec_[frame_index];
	}

	return true;
}

void FrameProviderPNG::skipNextFrame()
{
    if (file_iter_ >= files_.end()) return;
	file_iter_++;
}

void FrameProviderPNG::reset()
{
	file_iter_ = files_.begin();
}

bool FrameProviderPNG::getLastFramePose(Eigen::Affine3f & camera_pose_result)
{
	if (!expect_pose_ && camera_list_vec_.empty()) return false;
	camera_pose_result = last_frame_pose_;
	return true;
}


bool is_not_png_color_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	if (filename.string().find("depth") != std::string::npos) return true;
	return false; // is an oni png color image file
}
