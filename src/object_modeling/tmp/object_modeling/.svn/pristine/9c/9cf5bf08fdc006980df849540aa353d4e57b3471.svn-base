#pragma once

#include "frame_provider_base.h"

// hacky
#if defined (WIN32)
#undef max
#undef min
#endif
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>


class FrameProviderPCD : public FrameProviderBase
{
public:
	FrameProviderPCD(fs::path folder);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

	virtual bool skipNextFrame();

	virtual void reset();

protected:
	// members
	std::vector<fs::path> files;
	std::vector<fs::path>::iterator file_iter;
};

bool is_not_pcd_file(fs::path filename);