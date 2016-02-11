#pragma once

#include "frame_provider_base.h"

#include "params_camera.h"


class FrameProviderHanda : public FrameProviderBase
{
public:
	FrameProviderHanda(fs::path folder, ParamsCamera const& params_camera);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

protected:
	// members
	float depth_factor;
	std::vector<fs::path> files;
	std::vector<fs::path>::iterator file_iter;
	ParamsCamera params_camera;
};

bool is_not_handa_png_file(fs::path filename);
