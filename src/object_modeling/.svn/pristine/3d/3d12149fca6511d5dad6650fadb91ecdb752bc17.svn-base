#include "frame_provider_yuyin.h"

// for sorting #.jpg files
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/foreach.hpp>

FrameProviderYuyin::FrameProviderYuyin(fs::path folder, float depth_factor, std::string required_prefix)
	: depth_factor_(depth_factor)
{
	std::vector<fs::path> all_images;
	std::remove_copy_if(fs::directory_iterator(folder), fs::directory_iterator(), std::back_inserter(all_images), is_not_jpg_image);

	// sort the files by the number following required prefix
	typedef std::pair<int, fs::path> SortT;
	std::vector<SortT> temp_sort_vec;
	BOOST_FOREACH(fs::path const& f, all_images) {
		std::string num_string;
		int prefix_length = required_prefix.length();
		if (prefix_length > 0) {
			std::string f_prefix = f.stem().string().substr(0, prefix_length);
			if (f_prefix != required_prefix) continue;
			num_string = f.stem().string().substr(prefix_length);
		}
		else {
			num_string = f.stem().string();
		}
		int i = atoi(num_string.c_str());
		temp_sort_vec.push_back(std::make_pair(i, f));
	}
	std::sort(temp_sort_vec.begin(), temp_sort_vec.end());

	files_.clear();
	BOOST_FOREACH(SortT const& pair, temp_sort_vec) { 
		files_.push_back(pair.second);
	}

	file_iter_ = files_.begin();
	cout << "Found " << files_.size() << " frames" << endl;
}

bool FrameProviderYuyin::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter_ >= files_.end()) return false;

	// get file and increment
	fs::path filename = *file_iter_++;

	// assume #.png
	std::string frame_index_string = filename.stem().string();
	std::string expected_depth_filename = frame_index_string + ".png";
	// gotta replace "rgb" with "depth" in path
    fs::path rgb_path = fs::absolute(filename.parent_path());
	fs::path depth_path;
	for (fs::path::iterator iter = rgb_path.begin(); iter != rgb_path.end(); ++iter) {
		if (iter->string() == "processed") depth_path /= "processed_depth_big";
		else depth_path /= *iter;
	}
	cout << "rgb_path: " << rgb_path << endl;
	cout << "depth_path: " << depth_path << endl;
	fs::path expected_depth_file = depth_path / expected_depth_filename;

	// load the files
	cv::Mat color_bgr = cv::imread(filename.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	cv::Mat image_depth_png = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);
	// hack for wrong size...
	if (image_depth_png.rows + 2 == color.rows && image_depth_png.cols + 2 == color.cols) {
		cv::Mat depth_bigger(color.size(), CV_16UC1, cv::Scalar(0));
		cv::Mat roi = depth_bigger(cv::Rect(1,1,image_depth_png.cols,image_depth_png.rows));
		image_depth_png.copyTo(roi);
		image_depth_png = depth_bigger;
	}
	image_depth_png.convertTo(depth, CV_32F, 1./depth_factor_);

	return true;
}

void FrameProviderYuyin::skipNextFrame()
{
    if (file_iter_ >= files_.end()) return;
	file_iter_++;
}

void FrameProviderYuyin::reset()
{
	file_iter_ = files_.begin();
}


bool is_not_png_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	return false; // is a png file
}

bool is_not_jpg_image(fs::path filename)
{
	if (filename.extension() != ".jpg") return true;
	return false; // is a jpg file
}
