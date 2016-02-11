#include "keyframe_struct.h"

const float KeyframeStruct::depth_factor_save = 1000;

void KeyframeStruct::save(const fs::path & folder)
{
    if (!fs::exists(folder) && !fs::create_directories(folder)) {
        throw std::runtime_error("bad folder: " + folder.string());
    }

    fs::path filename = folder / "keyframe.yaml";
    cv::FileStorage fs(filename.string(), cv::FileStorage::WRITE);
    fs << "keypoints" << *keypoints;
    fs << "camera_index" << camera_index;
    std::vector<int> volumes_vec(volumes.begin(), volumes.end());
    fs << "volumes" << volumes_vec;
    fs.release();

    cv::imwrite( (folder / "mat_color_bgra.png").string(), mat_color_bgra);
    // this is a bit lossy!
    cv::Mat mat_depth_16u;
    mat_depth.convertTo(mat_depth_16u, CV_16U, depth_factor_save);
    cv::imwrite( (folder / "mat_depth_16u.png").string(), mat_depth_16u);
}

void KeyframeStruct::load(const fs::path & folder)
{
    if (!fs::exists(folder)) {
        throw std::runtime_error("bad folder: " + folder.string());
    }

    fs::path filename = folder / "keyframe.yaml";
    cv::FileStorage fs(filename.string(), cv::FileStorage::READ);
    keypoints.reset(new Keypoints);
    fs["keypoints"] >> *keypoints;
    fs["camera_index"] >> camera_index;
    std::vector<int> volumes_vec;
    fs["volumes"] >> volumes_vec;
    volumes = std::set<int>(volumes_vec.begin(), volumes_vec.end());

    // probably a flag to load bgra directly? CV_LOAD_IMAGE_ANYCOLOR didn't work...
    cv::Mat mat_color_bgr = cv::imread( (folder / "mat_color_bgra.png").string());
    cv::cvtColor(mat_color_bgr, mat_color_bgra, CV_BGR2BGRA);
    if (mat_color_bgra.type() != CV_8UC4) {
		cout << "bad image type! " << endl;
        throw std::runtime_error("bad image type");
    }
    
    cv::Mat mat_depth_16u = cv::imread( (folder / "mat_depth_16u.png").string(), CV_LOAD_IMAGE_ANYDEPTH);
    mat_depth_16u.convertTo(mat_depth, CV_32F, 1 / depth_factor_save);
}
