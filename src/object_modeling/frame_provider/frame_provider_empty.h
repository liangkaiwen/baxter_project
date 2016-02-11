#pragma once

#include "frame_provider_base.h"


class FrameProviderEmpty : public FrameProviderBase
{
public:
    FrameProviderEmpty();

    virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

    virtual void reset();
};
