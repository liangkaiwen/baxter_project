#pragma once

#include "typedefs.h"
#include "parameters.h"
#include "DisjointSet.h"

class MaskObject
{
public:
	MaskObject(const Parameters& params);

	bool addObjectCloudToFrame(FrameT& frame, const cv::Rect& seed_search_region);

	void fakeMasking(FrameT& frame); // to pretend that all valid points are in the mask (sets appropriate values in frame)

	void getObjectMask(const FrameT& frame, const cv::Rect& seedSearchRegion, int min_size, const cv::Mat& input_mask, bool debug_if_enabled, cv::Rect& object_rect, cv::Mat& object_mask);

	// I could make accessors for these....
	cv::Mat histogram_hand;
	cv::Mat histogram_object;

	void destroyWindows();

protected:
	void showInWindowToDestroy(const std::string& name, const cv::Mat& image);

	cv::Mat maskPartOfObject(const FrameT& frame, const cv::Mat& histogram);
	cv::Mat maskPartOfObjectAgainstBackground(const FrameT& frame, const cv::Mat& histogram_target, const cv::Mat& histogram_background);
	cv::Mat floodFillExpandMask(const FrameT& frame, const cv::Mat& floodFillSeedMask);
	cv::Mat removeSmallComponents(const FrameT& frame, const cv::Mat& input_mask, int min_component_area);
	cv::Mat removeSmallComponentsWithDisjointSet(const FrameT& frame, const cv::Mat& input_mask, int min_component_area);

	void depthMaskWithoutHand(const FrameT& frame, cv::Mat& mask_hand, cv::Mat& depth_mask_without_hand);

	int getObjectMaskComponent(const FrameT& frame, const cv::Rect& seedSearchRegion, const cv::Mat& input_mask, float input_max_depth, 
		cv::Rect& object_rect, cv::Mat& object_mask, float& min_depth, float& max_depth);

	

	const Parameters& params;
	std::set<std::string> windowsToDestroy;
};

