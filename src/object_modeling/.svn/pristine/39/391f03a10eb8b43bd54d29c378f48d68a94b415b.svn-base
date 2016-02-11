#include "frame_provider_handa.h"

#include <fstream>

FrameProviderHanda::FrameProviderHanda(fs::path folder, ParamsCamera const& params_camera)
	: params_camera(params_camera)
{
	std::remove_copy_if(fs::directory_iterator(folder), fs::directory_iterator(), std::back_inserter(files), is_not_handa_png_file);
	std::sort(files.begin(), files.end());
	file_iter = files.begin();
	cout << "Found " << files.size() << " frames" << endl;
}

bool FrameProviderHanda::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter >= files.end()) return false;

	// get file and increment
	fs::path filename = *file_iter++;

	// assume same stem + "depth"
	std::string expected_depth_filename = filename.stem().string() + ".depth";
	fs::path expected_depth_file = filename.parent_path() / (expected_depth_filename);

	if (!fs::exists(expected_depth_file)) {
		cout << "couldn't find: " << expected_depth_file << endl;
		return false;
	}

	cv::Mat color_bgr = cv::imread(filename.string());
	cv::cvtColor(color_bgr, color, CV_BGR2BGRA);

	// load the depth file
	// depths appear to be total distance from camera position, not just z coordinate

	depth = cv::Mat(color.size(), CV_32FC1);
	{
		std::ifstream ifs(expected_depth_file.c_str());
		for (int row = 0; row < depth.rows; ++row) {
			for (int col = 0; col < depth.cols; ++col) {
				float f;
				ifs >> f;
				// cos(arctan(x)) = 1 /sqrt(1 + x^2)
				float angle_row = atan( (row - params_camera.center.x()) / params_camera.focal.x());
				float angle_col = atan( (col - params_camera.center.y()) / params_camera.focal.y());
				float d = cos(angle_row)*cos(angle_col)*f;
				depth.at<float>(row,col) = d;
			}
		}
	}

	return true;
}

void FrameProviderHanda::skipNextFrame()
{
    if (file_iter >= files.end()) return;
	file_iter++;
}

void FrameProviderHanda::reset()
{
	file_iter = files.begin();
}


bool is_not_handa_png_file(fs::path filename)
{
	if (filename.extension() != ".png") return true;
	return false; // is a png file
}
