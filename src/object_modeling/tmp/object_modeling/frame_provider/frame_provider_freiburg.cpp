#include "frame_provider_freiburg.h"

#include <fstream>

#include <boost/algorithm/string.hpp>

FrameProviderFreiburg::FrameProviderFreiburg(fs::path associate_file)
{
#if 0
	1305031102.175304 rgb/1305031102.175304.png 1305031102.160407 depth/1305031102.160407.png
	1305031102.211214 rgb/1305031102.211214.png 1305031102.226738 depth/1305031102.226738.png
	1305031102.275326 rgb/1305031102.275326.png 1305031102.262886 depth/1305031102.262886.png
	1305031102.311267 rgb/1305031102.311267.png 1305031102.295279 depth/1305031102.295279.png
#endif

	fs::path absolute_path = fs::absolute(associate_file).parent_path();

	std::fstream file(associate_file.string().c_str(), std::ios::in);
	std::string s;
	while (std::getline(file, s)) {
		std::vector<std::string> tokens;
		boost::split(tokens, s, boost::is_any_of("\t "));
		if (tokens.size() != 4) break;
		frame_list_.push_back(FrameInformation());
		frame_list_.back().rgb_timestamp = ROSTimestamp(tokens[0]);
        frame_list_.back().rgb_path = absolute_path / boost::trim_copy(tokens[1]);
		frame_list_.back().depth_timestamp = ROSTimestamp(tokens[2]);
        frame_list_.back().depth_path = absolute_path / boost::trim_copy(tokens[3]);
	}
	frame_list_iterator_ = frame_list_.begin();
}

bool FrameProviderFreiburg::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (frame_list_iterator_ >= frame_list_.end()) return false;

	fs::path filename_rgb = frame_list_iterator_->rgb_path;
	fs::path filename_depth = frame_list_iterator_->depth_path;
	last_timestamp_ = frame_list_iterator_->rgb_timestamp; // use rgb for timestamps?  kind of arbitrary
	frame_list_iterator_++;

	cv::Mat color_bgr = cv::imread(filename_rgb.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	cv::Mat depth_png = cv::imread(filename_depth.string(), CV_LOAD_IMAGE_ANYDEPTH);
	const static float depth_factor = 5000.f;
	depth_png.convertTo(depth, CV_32F, 1./depth_factor);

	return true;
}

void FrameProviderFreiburg::skipNextFrame()
{
    if (frame_list_iterator_ >= frame_list_.end()) return;
	frame_list_iterator_++;
}

void FrameProviderFreiburg::reset()
{
	frame_list_iterator_ = frame_list_.begin();
}

bool FrameProviderFreiburg::getLastROSTimestamp(ROSTimestamp & ros_timestamp_result)
{
	ros_timestamp_result = last_timestamp_;
	return true;
}

