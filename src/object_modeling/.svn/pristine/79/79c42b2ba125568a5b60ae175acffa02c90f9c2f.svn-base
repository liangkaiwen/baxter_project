#include "frame_provider_pcd.h"

FrameProviderPCD::FrameProviderPCD(fs::path folder)
{
	std::remove_copy_if(fs::directory_iterator(folder), fs::directory_iterator(), std::back_inserter(files), is_not_pcd_file);
	std::sort(files.begin(), files.end());
	file_iter = files.begin();
	cout << "Found " << files.size() << " frames" << endl;
}

bool FrameProviderPCD::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	if (file_iter >= files.end()) return false;
	fs::path filename = *file_iter++;

	typedef pcl::PointXYZRGBA PointT;
	pcl::PointCloud<PointT>::Ptr cloud_ptr (new pcl::PointCloud<PointT>);
	if (pcl::io::loadPCDFile(filename.string(), *cloud_ptr)) {
		cout << "failed to load: " << filename << endl;
		return false;
	}

	int rows = cloud_ptr->height;
	int cols = cloud_ptr->width;
	cv::Mat result_color = cv::Mat(rows, cols, CV_8UC3);
	cv::Mat result_depth = cv::Mat(rows, cols, CV_32FC1);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			PointT& p = cloud_ptr->at(col, row);
			cv::Vec3b & c = result_color.at<cv::Vec3b>(row,col);
			c[0] = p.b;
			c[1] = p.g;
			c[2] = p.r;

			result_depth.at<float>(row,col) = p.z > 0 ? p.z : 0;
		}
	}

	color = result_color;
	depth = result_depth;

	return true;
}

bool FrameProviderPCD::skipNextFrame()
{
	if (file_iter >= files.end()) return false;
	file_iter++;
	return true;
}

void FrameProviderPCD::reset()
{
	file_iter = files.begin();
}




bool is_not_pcd_file(fs::path filename)
{
	if (filename.extension() != ".pcd") return true;
	return false;
}