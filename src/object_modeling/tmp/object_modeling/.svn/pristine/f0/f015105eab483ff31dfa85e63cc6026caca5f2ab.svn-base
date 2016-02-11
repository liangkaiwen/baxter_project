#include "frame_provider_luis.h"

#include "boost/format.hpp"

#include <algorithm>
#include <boost/algorithm/string.hpp>

FrameProviderLuis::FrameProviderLuis(fs::path folder, float depth_factor, fs::path luis_object_pose_txt)
    : FrameProviderBase(),
      depth_factor_(depth_factor)
{
	fs::path folder_color = folder / "color";
	std::remove_copy_if(fs::directory_iterator(folder_color), fs::directory_iterator(), std::back_inserter(files_), is_not_luis_color_image);
	std::sort(files_.begin(), files_.end());
	file_iter_ = files_.begin();
	cout << "Found " << files_.size() << " frames" << endl;

    // we will overwrite camera_list_vec_ in the base class with luis poses
    if (!luis_object_pose_txt.empty()) {
/*
%time,field.position.x,field.position.y,field.position.z,field.orientation.x,field.orientation.y,field.orientation.z,field.orientation.w
1423080941837986635,0.00484555214643,0.0319808498025,1.01757240295,0.598677515984,0.646155416965,-0.320760607719,0.348110586405
1423080941977079082,0.00484485551715,0.0319762527943,1.01742613316,0.591733753681,0.65271115303,-0.324425816536,0.344335764647
1423080942165721895,0.00484488811344,0.0319764688611,1.01743292809,0.585968255997,0.657987713814,-0.327448636293,0.34127804637
		*/

        fs::path filename = folder / luis_object_pose_txt;
		std::fstream file(filename.string().c_str(), std::ios::in);
		std::vector<std::string> lines;

		std::copy(std::istream_iterator<std::string>(file),
			std::istream_iterator<std::string>(),
			std::back_inserter(lines));

		camera_list_vec_.clear();
		for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); ++iter) {
			if (iter->empty()) continue;
			if (iter->at(0) == '%') continue;
			//std::string spaced = boost::replace_all_copy(*iter, ",", " ");
			std::vector<std::string> split;
            std::string trimmed = boost::trim_copy(*iter);
            boost::split(split, trimmed, boost::is_any_of(","));
			if (split.size() != 8) {
				cout << "bad split size" << endl;
				exit(1);
			}

			// expects: tx ty tz qw qx qy qz
			std::string transform_string = (boost::format("%s %s %s %s %s %s %s") % split[1] % split[2] % split[3] % split[7] % split[4] % split[5] % split[6]).str();
			// note inverse here (luis is object poses);
            camera_list_vec_.push_back(PosePtr(new Eigen::Affine3f(EigenUtilities::stringToTransform(transform_string).inverse())));
		}

		if (camera_list_vec_.size() != files_.size()) {
			cout << "camera_list_vec_.size() != files_.size()" << endl;
			exit(1);
		}
	}
}

bool FrameProviderLuis::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter_ >= files_.end()) return false;

	// get file and increment
	fs::path filename = *file_iter_++;

	// assume ../depth/(samename)
	std::string frame_index_string = filename.stem().string().substr(5,3); // assumes "frame001.png"
	//std::string expected_depth_filename = (boost::format("frame%03d") % frame_index_string).str() + ".png";
	std::string expected_depth_filename = filename.filename().string();
	fs::path expected_depth_file = filename.parent_path().parent_path() / "depth" / expected_depth_filename;

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find expected_depth_file: " << expected_depth_file << endl;
		return false;
	}

	cv::Mat color_bgr = cv::imread(filename.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	cv::Mat image_depth_png = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);
	image_depth_png.convertTo(depth, CV_32F, 1./depth_factor_);

	if (!camera_list_vec_.empty()) {
		int frame_index = atoi(frame_index_string.c_str());
        last_frame_pose_ = *camera_list_vec_[frame_index];
	}

	return true;
}

void FrameProviderLuis::skipNextFrame()
{
    if (file_iter_ >= files_.end()) return;
	file_iter_++;
}

void FrameProviderLuis::reset()
{
	file_iter_ = files_.begin();
}

bool FrameProviderLuis::getLastFramePose(Eigen::Affine3f & camera_pose_result)
{
	if (camera_list_vec_.empty()) return false;
	camera_pose_result = last_frame_pose_;
	return true;
}


bool is_not_luis_color_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	// not really needed for luis because folder:
	if (filename.string().find("depth") != std::string::npos) return true;
	return false; // is an oni png color image file
}
