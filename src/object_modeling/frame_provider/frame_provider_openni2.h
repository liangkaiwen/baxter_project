#pragma once

#include "frame_provider_base.h"

#include <openni2/OpenNI.h>


struct FrameProviderOpenni2Params
{
	bool auto_white_balance;
	bool auto_exposure;
	fs::path file;
	int resolution_x;
	int resolution_y;
	int fps;

	FrameProviderOpenni2Params() :
		auto_white_balance(true),
		auto_exposure(true),
		resolution_x(640),
		resolution_y(480),
		fps(30)
		{}
};

class FrameProviderOpenni2 : public FrameProviderBase
{
public:
    FrameProviderOpenni2(FrameProviderOpenni2Params & params);
	virtual ~FrameProviderOpenni2();

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	void setAutoExposure(bool value);
	void setAutoWhiteBalance(bool value);

	void setRecording(bool value);

protected:

	void init();

	bool getNextFrameRefs(openni::VideoFrameRef & color_frame_ref, openni::VideoFrameRef & depth_frame_ref);

	std::string getFreshFilename();

	openni::Device device;
	openni::VideoStream color_stream;
	openni::VideoStream depth_stream;

	FrameProviderOpenni2Params params;

	int frame_count;
	int frame_counter;

	// make this work for recording too
	boost::shared_ptr<openni::Recorder> recorder_ptr;
};
