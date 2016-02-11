#include "keypoints.h"

void Keypoints::write(cv::FileStorage & fs) const
{
	if (keypoints.empty()) return; // nothing to do if empty;

	fs << "{";

	fs << "keypoints" << keypoints;
	fs << "descriptors" << descriptors;
	cv::Mat points_cv;
	cv::eigen2cv(points, points_cv);
	fs << "points_cv" << points_cv;

	fs << "}";
}

void Keypoints::read(const cv::FileNode & node)
{
#if 1
	cv::read(node["keypoints"], keypoints);
	cv::read(node["descriptors"], descriptors);
	cv::Mat points_cv;
	cv::read(node["points_cv"], points_cv);
	points.resize(4, points_cv.cols);

	//	cv::cv2eigen(points_cv, points);
	//cout << "THIS IS STUPID (in keypoints.cpp)" << endl;

	// note: this extra copy seems to be required...it's just a load, so who cares...
	Eigen::MatrixXf points_temp(4, points_cv.cols);
	cv::cv2eigen(points_cv, points_temp);
	points = points_temp;
#else
	// this doesn't work for some reason:
	node["keypoints"] >> keypoints;
	node["descriptors"] >> descriptors;
	cv::Mat points_cv;
	node["points_cv"] >> points_cv;
	cv::cv2eigen(points_cv, points);
#endif
}

namespace cv {
void write(cv::FileStorage& fs, const std::string&, const Keypoints& x)
{
	x.write(fs);
}

void read(const cv::FileNode& node, Keypoints& x, const Keypoints& default_value)
{
	if (node.empty()) {
		cout << "Keypoints FileNode read: EMPTY NODE!!" << endl;
		x = default_value;
	}
	else {
		x.read(node);
	}
}
} // ns