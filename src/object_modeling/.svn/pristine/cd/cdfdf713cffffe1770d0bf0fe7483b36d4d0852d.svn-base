#include "frame_provider_file_wrapper.h"

FrameProviderFileWrapper::FrameProviderFileWrapper(boost::shared_ptr<FrameProviderBase> frame_provider_ptr, int frame_increment, int frame_start, int frame_end)
	: frame_increment(frame_increment),
	frame_start(frame_start),
	frame_end(frame_end),
    frame_provider_ptr(frame_provider_ptr),
    frame_counter_actual(0)
{
    if (frame_increment < 1) {
        cout << "Illegal value for frame_increment: " << frame_increment << endl;
        cout << "Setting frame_increment to 1" << endl;
        frame_increment = 1;
    }

	reset();
}

bool FrameProviderFileWrapper::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
    // we skip the first frame_increment-1 frames
    for (int i = 0; i < frame_increment-1; ++i) {
        frame_provider_ptr->skipNextFrame();
    }

	// note the ++ in here for the getNextFrame
    if (frame_end > 0 && frame_counter_actual++ >= frame_end) return false;

	return frame_provider_ptr->getNextFrame(color, depth);
}

void FrameProviderFileWrapper::skipNextFrame()
{
    for (int i = 0; i < frame_increment; ++i) {
        frame_provider_ptr->skipNextFrame();
    }
}

void FrameProviderFileWrapper::reset()
{
	frame_provider_ptr->reset();
    frame_counter_actual = 0;


    for (int i = 0; i < frame_start; ++i) {
        this->skipNextFrame();
    }
}

bool FrameProviderFileWrapper::getLastFramePose(Eigen::Affine3f & camera_pose_result)
{
	return frame_provider_ptr->getLastFramePose(camera_pose_result);
}

bool FrameProviderFileWrapper::getLastROSTimestamp(ROSTimestamp & ros_timestamp_result)
{
	return frame_provider_ptr->getLastROSTimestamp(ros_timestamp_result);
}

bool FrameProviderFileWrapper::getLastFilename(std::string & result_filename)
{
    return frame_provider_ptr->getLastFilename(result_filename);
}
