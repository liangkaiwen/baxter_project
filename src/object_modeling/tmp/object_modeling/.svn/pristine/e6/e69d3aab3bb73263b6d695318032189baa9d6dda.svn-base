#include "frame_provider_png_depth.h"

#include "EigenUtilities.h"

#include <boost/foreach.hpp>

FrameProviderPNGDepth::FrameProviderPNGDepth(fs::path folder, float depth_factor, fs::path camera_list_txt, cv::Scalar fake_color)
    : depth_factor_(depth_factor),
	fake_color_(fake_color)
{
    std::remove_copy_if(fs::directory_iterator(folder), fs::directory_iterator(), std::back_inserter(files_), is_not_png_depth_image);
	std::sort(files_.begin(), files_.end());
	file_iter_ = files_.begin();
	cout << "Found " << files_.size() << " frames" << endl;

	if (!camera_list_txt.empty()) {
        //	void loadPosesFromFile(fs::path filename, PosePtrList & result);
        // just expect standard pose file
        PosePtrList pose_ptr_list;
        EigenUtilities::loadPosesFromFile(folder / camera_list_txt, pose_ptr_list);
        BOOST_FOREACH(PosePtr & p, pose_ptr_list) {
            camera_list_vec_.push_back(*p);
        }

        cout << "Found " << camera_list_vec_.size() << " camera poses" << endl;
	}
}

bool FrameProviderPNGDepth::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter_ >= files_.end()) return false;

	// get file and increment
	fs::path filename = *file_iter_++;

    // expect "depth_#####.png"
    std::string frame_index_string = filename.stem().string().substr(6,5);

    // here the filename is the depth file, and we create a fake color image of the same size
    cv::Mat image_depth_png = cv::imread(filename.string(), CV_LOAD_IMAGE_ANYDEPTH);
	image_depth_png.convertTo(depth, CV_32F, 1./depth_factor_);

    color = cv::Mat(image_depth_png.size(), CV_8UC4, fake_color_);

    if (!camera_list_vec_.empty()) {
		int frame_index = atoi(frame_index_string.c_str());
		last_frame_pose_ = camera_list_vec_[frame_index];
	}

	return true;
}

void FrameProviderPNGDepth::skipNextFrame()
{
    if (file_iter_ >= files_.end()) return;
    file_iter_++;
}

void FrameProviderPNGDepth::reset()
{
	file_iter_ = files_.begin();
}

bool FrameProviderPNGDepth::getLastFramePose(Eigen::Affine3f & camera_pose_result)
{
    if (camera_list_vec_.empty()) return false;
	camera_pose_result = last_frame_pose_;
	return true;
}


bool is_not_png_depth_image(fs::path filename)
{
	if (filename.extension() != ".png") return true;
    if (filename.string().find("color") != std::string::npos) return true;
    return false;
}
