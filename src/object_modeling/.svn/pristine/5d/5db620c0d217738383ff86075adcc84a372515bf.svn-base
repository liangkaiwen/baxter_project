#pragma once

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <opencv_utilities.h>

#include "keypoints.h"

struct KeyframeStruct {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    const static float depth_factor_save;

    cv::Mat mat_color_bgra;
    cv::Mat mat_depth;
    boost::shared_ptr<Keypoints> keypoints;
    int camera_index;
    std::set<int> volumes; // volumes seen by keyframe

    void save(const fs::path & folder);
    void load(const fs::path & folder);


};
typedef boost::shared_ptr<KeyframeStruct> KeyframeStructPtr;
