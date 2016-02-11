#pragma once

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <stdexcept>
#include <iostream>
using std::cout;
using std::endl;
#include <string>
#include <algorithm>

#include "EigenUtilities.h"
#include "ros_timestamp.h"

class FrameProviderBase
{
public:
    FrameProviderBase();

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth) = 0;

    virtual void skipNextFrame() = 0;

	virtual void reset() = 0;

	// a bit hacked in...can call this after true calls to getNextFrame
	// just returns false if no pose available
	virtual bool getLastFramePose(Eigen::Affine3f & camera_pose_result);

	// also hacked in...some frame providers have timestamps (specifically Freiburg)
	virtual bool getLastROSTimestamp(ROSTimestamp & ros_timestamp_result);

    // and more hacking... get the last filename to associate poses with filenames (not full paths)
    virtual bool getLastFilename(std::string & result_filename);
};
