#pragma once

#include "params_mask_object.h"

#include "frame.h"

#include "histogram_util.h"
#include "learn_histogram.h"

#include "basic.h" // really?

class MaskObject
{
public:
	MaskObject(const ParamsMaskObject & params_mask_object);

	void setHistogramHand(const cv::Mat & histogram_hand);
	void setHistogramObject(const cv::Mat & histogram_object);
	cv::Mat getHistogramHand() const;
	cv::Mat getHistogramObject() const;

	void resetSeedSearchRegion();

	void save(fs::path const& folder);
	void load(fs::path const& folder);

	void getObjectMask(const Frame& frame, cv::Rect & object_rect, cv::Mat & object_mask);

	std::map<std::string, cv::Mat> const& getDebugImages() const;

protected:
	// functions
	void getObjectMask(const Frame & frame, const cv::Mat & input_mask, cv::Rect & object_rect, cv::Mat & object_mask);

	// idea: can use this (combined with a depth mask that you get elsewhere) to get input_mask for getObjectMask
	void maskHand(const Frame & frame, cv::Mat & mask_hand);

	int getObjectMaskComponent(const Frame & frame, const cv::Mat & input_mask, float input_max_depth, cv::Rect & object_rect, cv::Mat & object_mask, float & min_depth, float & max_depth);

	cv::Mat maskHandWithHistograms(const cv::Mat & mat_color_hsv);

	cv::Mat removeSmallComponentsWithDisjointSet(const cv::Mat& input_mask, int min_component_area);

	cv::Mat floodFillExpandMask(const cv::Mat & image_color, const cv::Mat& input_mask);


	// members

	// no save
	ParamsMaskObject params_mask_object_;

	// assume loaded from files (no save?)
	cv::Mat histogram_hand_;
	cv::Mat histogram_object_;

	std::map<std::string, cv::Mat> debug_images_;

	// save
	cv::Rect seed_search_region_;

};

