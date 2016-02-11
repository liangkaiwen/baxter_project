#include "frame_provider_arun.h"

#include <boost/foreach.hpp>

// assume:
// ./color/frame###.png
// ./depth/frame###.png

FrameProviderArun::FrameProviderArun(fs::path folder, float depth_factor)
	: depth_factor_(depth_factor)
{
	std::remove_copy_if(fs::directory_iterator(folder / "color"), fs::directory_iterator(), std::back_inserter(files_), FrameProviderArun::is_not_png_image);
    std::sort(files_.begin(), files_.end());

	file_iter_ = files_.begin();
	cout << "Found " << files_.size() << " frames" << endl;
}

bool FrameProviderArun::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter_ >= files_.end()) return false;

	// get file and increment
	fs::path filename_color = *file_iter_++;

    // need ../depth/filename_color for depth
    fs::path path_color = fs::absolute(filename_color.parent_path());
    fs::path path_depth = path_color / ".." / "depth";
    fs::path filename_depth = path_depth / filename_color.filename();
    // if I don't see a depth file, just skip to the next color image and use that ;)
    if (!fs::exists(filename_depth)) {
        cout << "[Arun] Didn't find expected depth: " << filename_depth << endl;
        return getNextFrame(color, depth);
    }
#if 0
    if (filename_color.filename().string().size() != 12) {
        cout << "[Arun] Ignoring filename of wrong length: " << filename_color.filename() << endl;
        return getNextFrame(color, depth);
    }
#endif

	// load the files
    cv::Mat color_bgr = cv::imread(filename_color.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	cv::Mat image_depth_png = cv::imread(filename_depth.string(), CV_LOAD_IMAGE_ANYDEPTH);
	image_depth_png.convertTo(depth, CV_32F, 1./depth_factor_);

    // set the last filename to go with these
    last_color_file_path_ = filename_color;

	return true;
}

void FrameProviderArun::skipNextFrame()
{
    if (file_iter_ >= files_.end()) return;
	file_iter_++;
}

void FrameProviderArun::reset()
{
	file_iter_ = files_.begin();
}

bool FrameProviderArun::getLastFilename(std::string &result_filename)
{
    if (!last_color_file_path_.empty()) {
        result_filename = last_color_file_path_.filename().string();
        return true;
    }
    return false;
}


bool FrameProviderArun::is_not_png_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	return false; // is a png file
}

