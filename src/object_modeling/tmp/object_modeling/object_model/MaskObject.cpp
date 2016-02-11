#include "stdafx.h"
#include "MaskObject.h"

#include "histogramUtil.h"
#include "opencvUtil.h"

using namespace std;

MaskObject::MaskObject(const Parameters& params)
	: params(params)
{
}

void MaskObject::depthMaskWithoutHand(const FrameT& frame, cv::Mat& mask_hand, cv::Mat& depth_mask_without_hand)
{
	mask_hand = maskPartOfObjectAgainstBackground(frame, histogram_hand, histogram_object);

	if (params.mask_debug_images) {
		showInWindowToDestroy("Mask hand: before everything", mask_hand);
	}

	if (params.mask_hand_min_component_area_before_morphology > 0) {
		//mask_hand = removeSmallComponents(frame, mask_hand, params.mask_hand_min_component_area_before_morphology);
		mask_hand = removeSmallComponentsWithDisjointSet(frame, mask_hand, params.mask_hand_min_component_area_before_morphology);

		if (params.mask_debug_images) {
			showInWindowToDestroy("Mask hand: after removeSmallComponentsWithDisjointSet before morphology", mask_hand);
		}
	}

	// erode?
	for (int i = 0; i < params.mask_hand_erode; i++) cv::erode(mask_hand, mask_hand, cv::Mat());

	// dilate?
	for (int i = 0; i < params.mask_hand_dilate; i++) cv::dilate(mask_hand, mask_hand, cv::Mat());

	if (params.mask_hand_min_component_area_after_morphology > 0) {
		//mask_hand = removeSmallComponents(frame, mask_hand, params.mask_hand_min_component_area_after_morphology);
		mask_hand = removeSmallComponentsWithDisjointSet(frame, mask_hand, params.mask_hand_min_component_area_after_morphology);

		if (params.mask_debug_images) {
			showInWindowToDestroy("Mask hand: after removeSmallComponentsWithDisjointSet after morphology", mask_hand);
		}
	}

	if (params.mask_floodfill) {
		mask_hand = floodFillExpandMask(frame, mask_hand);
		showInWindowToDestroy("Mask hand: after floodFillExpandMask", mask_hand);
	}

	cv::Mat mask_not_hand;
	cv::bitwise_not(mask_hand, mask_not_hand);
	cv::bitwise_and(frame.depth_mask, mask_not_hand, depth_mask_without_hand);
}

cv::Mat MaskObject::maskPartOfObject(const FrameT& frame, const cv::Mat& histogram)
{
	if (histogram.dims != 3) throw std::runtime_error ("histogram.dims != 3");

	cv::Mat back_project;
	cv::calcBackProject(&frame.image_color_hsv, 1, HistogramConstants::hist_hsv_channels, histogram, back_project, HistogramConstants::hist_hsv_ranges, 1.0);

	if (params.mask_debug_images) showInWindowToDestroy("Backproject", back_project);

	cv::Mat back_project_mask;
	float threshold_for_bytes = params.mask_hand_backproject_thresh * 255;
	cv::threshold(back_project, back_project_mask, threshold_for_bytes, 255, CV_THRESH_BINARY);

	if (params.mask_debug_images) showInWindowToDestroy("Backproject Mask", back_project_mask);

	return back_project_mask;
}

cv::Mat MaskObject::maskPartOfObjectAgainstBackground(const FrameT& frame, const cv::Mat& histogram_target, const cv::Mat& histogram_background)
{
	if (histogram_target.dims != 3) throw std::runtime_error ("histogram_target.dims != 3");
	if (histogram_background.dims != 3) throw std::runtime_error ("histogram_background.dims != 3");

	cv::Mat back_project_target;
	cv::calcBackProject(&frame.image_color_hsv, 1, HistogramConstants::hist_hsv_channels, histogram_target, back_project_target, HistogramConstants::hist_hsv_ranges, 1.0);
	cv::Mat back_project_background;
	cv::calcBackProject(&frame.image_color_hsv, 1, HistogramConstants::hist_hsv_channels, histogram_background, back_project_background, HistogramConstants::hist_hsv_ranges, 1.0);

	// do we really need floats here?
	back_project_target.convertTo(back_project_target, CV_32F, 1./255);
	back_project_background.convertTo(back_project_background, CV_32F, 1./255);

	//if (params.mask_debug_images) showInWindowToDestroy("Backproject Target", back_project_target);
	//if (params.mask_debug_images) showInWindowToDestroy("Backproject Background", back_project_background);

	cv::Mat back_project_sum = back_project_target + back_project_background;
	//if (params.mask_debug_images) showInWindowToDestroy("Backproject Sum", back_project_sum);

	cv::Mat back_project_relative = back_project_target / back_project_sum;
	if (params.mask_debug_images) showInWindowToDestroy("Backproject Relative", back_project_relative);

	cv::Mat back_project_mask;
	//float threshold_for_bytes = params.mask_hand_backproject_thresh * 255;
	//cv::threshold(back_project_relative, back_project_mask, threshold_for_bytes, 255, CV_THRESH_BINARY);
	// because floats:
	cv::threshold(back_project_relative, back_project_mask, params.mask_hand_backproject_thresh, 255, CV_THRESH_BINARY);

	back_project_mask.convertTo(back_project_mask, CV_8U);

	if (params.mask_debug_images) showInWindowToDestroy("Backproject Mask", back_project_mask);

	return back_project_mask;

}

cv::Mat MaskObject::removeSmallComponents(const FrameT& frame, const cv::Mat& input_mask, int min_component_area)
{
	if (min_component_area <= 0) return input_mask;

	cv::Mat input_mask_mutable = input_mask.clone();
	const static int connectivity = 8;
	const static int floodfill_mask_set = 255;
	const static int flags = connectivity + (floodfill_mask_set << 8) + cv::FLOODFILL_MASK_ONLY;
	const cv::Scalar lowDiff = cv::Scalar::all(0);
	const cv::Scalar highDiff = cv::Scalar::all(0);
	cv::Mat result(input_mask.size(), input_mask.type(), cv::Scalar::all(0));
	for (int row = 0; row < input_mask_mutable.rows; row++) {
		for (int col = 0; col < input_mask_mutable.cols; col++) {
			if (input_mask_mutable.at<unsigned char>(row, col)) {
				// keep collected mask of components over area
				cv::Mat thisFloodMask(input_mask.size() + cv::Size(2,2), CV_8UC1, cv::Scalar(0));
				cv::Mat thisFloodMaskImageSize = thisFloodMask(cv::Rect(1,1,input_mask.cols,input_mask.rows));
				int area = cv::floodFill(input_mask_mutable, thisFloodMask, cv::Point(col, row), 0, NULL, lowDiff, highDiff, flags);
				//cout << "area: " << area << endl;
				cv::Mat points_to_remove;
				cv::bitwise_not(thisFloodMaskImageSize, points_to_remove);
				cv::bitwise_and(input_mask_mutable, points_to_remove, input_mask_mutable);
				if (area > min_component_area) {
					cv::bitwise_or(result, thisFloodMaskImageSize, result);
				}
			}
		}
	}

	if (params.mask_debug_images) {
		showInWindow("removeSmallComponents input", input_mask);
		showInWindow("removeSmallComponents result", result);
	}

	return result;
}

cv::Mat MaskObject::removeSmallComponentsWithDisjointSet(const FrameT& frame, const cv::Mat& input_mask, int min_component_area)
{
	if (min_component_area <= 0) return input_mask; // note this is not a deep copy

	DisjointSet disjoint_set(input_mask.cols * input_mask.rows);

	// union right
	for (int row = 0; row < input_mask.rows; row++) {
		for (int col = 0; col < input_mask.cols-1; col++) {
			int index_1 = row * input_mask.cols + col;
			int index_2 = row * input_mask.cols + (col + 1);
			if (input_mask.data[index_1] && input_mask.data[index_2]) disjoint_set.connect(index_1, index_2);
		}
	}

	// union right-down
	for (int row = 0; row < input_mask.rows-1; row++) {
		for (int col = 0; col < input_mask.cols-1; col++) {
			int index_1 = row * input_mask.cols + col;
			int index_2 = (row+1) * input_mask.cols + (col + 1);
			if (input_mask.data[index_1] && input_mask.data[index_2]) disjoint_set.connect(index_1, index_2);
		}
	}

	// union down
	for (int row = 0; row < input_mask.rows-1; row++) {
		for (int col = 0; col < input_mask.cols; col++) {
			int index_1 = row * input_mask.cols + col;
			int index_2 = (row+1) * input_mask.cols + col;
			if (input_mask.data[index_1] && input_mask.data[index_2]) disjoint_set.connect(index_1, index_2);
		}
	}

	// union down-left
	for (int row = 0; row < input_mask.rows-1; row++) {
		for (int col = 1; col < input_mask.cols; col++) {
			int index_1 = row * input_mask.cols + col;
			int index_2 = (row+1) * input_mask.cols + col - 1;
			if (input_mask.data[index_1] && input_mask.data[index_2]) disjoint_set.connect(index_1, index_2);
		}
	}

	cv::Mat result(input_mask.size(), input_mask.type(), cv::Scalar::all(0));
	for (int row = 0; row < input_mask.rows; row++) {
		for (int col = 0; col < input_mask.cols; col++) {
			int index = row * input_mask.cols + col;
			if (input_mask.data[index] && disjoint_set.size(index) >= min_component_area) result.data[index] = 255;
		}
	}

	return result;
}

cv::Mat MaskObject::floodFillExpandMask(const FrameT& frame, const cv::Mat& input_mask)
{
	// set up accumulatedFloodMask
	const static int connectivity = 8;
	const static int hand_mask_value = 255;
	const static int non_object_mask_value = 100; // not used?
	cv::Mat accumulatedFloodMask(input_mask.size() + cv::Size(2,2), CV_8UC1, cv::Scalar(0));
	cv::Mat accumulatedFloodMaskImageSize = accumulatedFloodMask(cv::Rect(1,1,input_mask.cols,input_mask.rows));

	//input_mask.copyTo(accumulatedFloodMaskImageSize);

	cv::Mat imageForFloodFill = frame.image_color.clone(); // clone necessary?
	const cv::Scalar lowDiff(params.mask_floodfill_expand_diff, params.mask_floodfill_expand_diff, params.mask_floodfill_expand_diff);
	const cv::Scalar highDiff(params.mask_floodfill_expand_diff, params.mask_floodfill_expand_diff, params.mask_floodfill_expand_diff);
	const static int flags = connectivity + (hand_mask_value << 8) + cv::FLOODFILL_MASK_ONLY;
	int area_sum = 0;
	for (int row = 0; row < imageForFloodFill.rows; row++) {
		for (int col = 0; col < imageForFloodFill.cols; col++) {
			if (!input_mask.at<unsigned char>(row, col)) continue;

			//if (accumulatedFloodMaskImageSize.at<unsigned char>(row, col)) continue;

			//cv::Rect floodFillRectResult; // could get this instead of NULL
			int area = cv::floodFill(imageForFloodFill, accumulatedFloodMask, cv::Point(col, row), 0, NULL, lowDiff, highDiff, flags);
			area_sum += area;
			//cout << "area: " << area << endl;
			//showInWindowToDestroy("Accumulated Flood Mask (in loop)", accumulatedFloodMask);
			//cv::waitKey();
		}
	}

	cv::Mat flood_fill_mask_result; // result
	cv::threshold(accumulatedFloodMaskImageSize, flood_fill_mask_result, non_object_mask_value+1, 255, CV_THRESH_BINARY);

	return flood_fill_mask_result;
}


void MaskObject::showInWindowToDestroy(const std::string& name, const cv::Mat& image)
{
	windowsToDestroy.insert(name);
	showInWindow(name, image);
}

void MaskObject::destroyWindows()
{
	for (std::set<std::string>::iterator i = windowsToDestroy.begin(); i != windowsToDestroy.end(); ++i) {
		cv::destroyWindow(*i);
	}
}

int MaskObject::getObjectMaskComponent(const FrameT& frame, const cv::Rect& seedSearchRegion, const cv::Mat& input_mask, float input_max_depth, 
	cv::Rect& object_rect, cv::Mat& object_mask, float& min_depth, float& max_depth)
{
	// find the closest point within seedSearchRegion
	cv::Mat depth_image_roi = frame.image_depth(seedSearchRegion);
	double minVal;
	cv::Point minLoc;
	cv::Mat mask_roi = input_mask(seedSearchRegion);

	if (cv::countNonZero(mask_roi) == 0) {
		object_rect = cv::Rect();
		object_mask = cv::Mat();
		return 0;
	}

	cv::minMaxLoc(depth_image_roi, &minVal, NULL, &minLoc, NULL, mask_roi);
	cv::Point seedPoint = seedSearchRegion.tl() + minLoc;

	if (input_max_depth > 0 && minVal > input_max_depth) {
		object_rect = cv::Rect();
		object_mask = cv::Mat();
		return 0;
	}

	// floodFill for depth masking
	cv::Mat depthForFloodFill = cv::Mat::zeros(frame.image_depth.size(), frame.image_depth.type()); // later assuming CV_32FC1
	frame.image_depth.copyTo(depthForFloodFill, input_mask);
	cv::Mat floodFillMask = cv::Mat::zeros(frame.image_depth.size() + cv::Size(2,2), CV_8UC1);
	cv::Rect floodRect;
	int connectivity = 8;
	int newMaskVal = 255;
	int flags = connectivity + (newMaskVal << 8) + cv::FLOODFILL_MASK_ONLY;
	int area = cv::floodFill(depthForFloodFill, floodFillMask, seedPoint, 0, &floodRect, cv::Scalar(params.mask_connected_max_depth_difference), cv::Scalar(params.mask_connected_max_depth_difference), flags);

	object_rect = floodRect;
	object_mask = floodFillMask (cv::Rect(1,1,frame.image_depth.cols,frame.image_depth.rows)); // maybe clone to get rid of extra data?

	double min_depth_double;
	double max_depth_double;
	cv::minMaxLoc(frame.image_depth, &min_depth_double, &max_depth_double, NULL, NULL, object_mask);
	min_depth = min_depth_double;
	max_depth = max_depth_double;

	return area;
}

void MaskObject::getObjectMask(const FrameT& frame, const cv::Rect& seedSearchRegion, int min_size, const cv::Mat& input_mask, bool debug_if_enabled, cv::Rect& object_rect, cv::Mat& object_mask)
{
	cv::Mat input_mask_mutable = input_mask.clone();
	float max_depth_input = -1; // negative means ignore
	float min_found_depth = 100;

	object_rect = cv::Rect();
	object_mask = cv::Mat::zeros(input_mask.size(), CV_8UC1);
	while(true) {
		cv::Rect this_object_rect;
		cv::Mat this_object_mask;
		float min_depth;
		float max_depth;
		int component_area = getObjectMaskComponent(frame, seedSearchRegion, input_mask_mutable, max_depth_input, this_object_rect, this_object_mask, min_depth, max_depth);
		if (this_object_rect == cv::Rect()) break;
		// is this check necessary?
		if (object_rect == cv::Rect()) object_rect = this_object_rect;
		else object_rect |= this_object_rect;
		cv::bitwise_or(object_mask, this_object_mask, object_mask);
		input_mask_mutable.setTo(0, object_mask);
		max_depth_input = max(max_depth_input, max_depth + params.mask_disconnected_max_depth_difference);

		// use min_found_depth over all components to constraint max_depth_input as well
		min_found_depth = min(min_found_depth, min_depth);
		max_depth_input = min(max_depth_input, min_found_depth + params.mask_global_max_depth_difference);

		if (debug_if_enabled && params.mask_debug_every_component) {
			showInWindow("mask_debug_every_component", this_object_mask);
			cv::Mat component_color(frame.image_color.size(), frame.image_color.type(), cv::Scalar(0,0,255));
			frame.image_color.copyTo(component_color, this_object_mask);
			showInWindow("mask_debug_every_component (color)", component_color);
			cout << "mask_debug_every_component area: " << component_area << endl;
			cv::waitKey();
		}

		if (params.mask_object_use_only_first_segment) break;
	}

	// could put this check inside if have spurious near small points??
	if (object_rect.width < min_size || object_rect.height < min_size) {
		object_rect = cv::Rect();
		object_mask = cv::Mat::zeros(input_mask.size(), CV_8UC1);
	}

	// erode the mask?
	for (int i = 0; i < params.mask_object_erode; i++) cv::erode(object_mask, object_mask, cv::Mat());
}

// also adds object_mask and object_rect
bool MaskObject::addObjectCloudToFrame(FrameT& frame, const cv::Rect& seed_search_region)
{
	CloudT::Ptr result(new CloudT);

	// create frame.depth_mask_without_hand
	if (params.mask_hand) {
		cv::Mat mask_hand; // can see this out here now
		depthMaskWithoutHand(frame, mask_hand, frame.depth_mask_without_hand);

		// debug
		if (params.mask_debug_images) {
			showInWindow("mask_hand", mask_hand);
		}
	}
	else {
		frame.depth_mask_without_hand = frame.depth_mask;
	}

	// extract the actual object mask once hand pixels have been masked out
	getObjectMask(frame, seed_search_region, params.mask_object_min_size, frame.depth_mask_without_hand, true, frame.object_rect, frame.object_mask);

	// extract an organized subcloud
	// if no mask was found, the object_rect should be empty
	result->header = frame.cloud_ptr->header;
	result->is_dense = false;
	result->sensor_origin_ = frame.cloud_ptr->sensor_origin_;
	result->sensor_orientation_ = frame.cloud_ptr->sensor_orientation_;

	result->width = frame.object_rect.width;
	result->height = frame.object_rect.height;
	result->points.resize(result->width * result->height);

	for (int row = 0; row < frame.object_rect.height; row++) {
		for (int col = 0; col < frame.object_rect.width; col++) {
			int row_full = row + frame.object_rect.y;
			int col_full = col + frame.object_rect.x;
			PointT& p_r = result->at(col, row);
			p_r = frame.cloud_ptr->at(col_full, row_full);
			uchar mask_value = frame.object_mask.at<uchar>(row_full, col_full);
			if (!mask_value) {
				p_r.x = p_r.y = p_r.z = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}

	frame.object_cloud_ptr = result;

	return (frame.object_rect != cv::Rect());
}

/*
frame.object_rect
frame.object_mask
frame.depth_mask_without_hand
frame.object_cloud_ptr
*/
void MaskObject::fakeMasking(FrameT& frame)
{
	int rows = frame.cloud_ptr->height;
	int cols = frame.cloud_ptr->width;
	frame.depth_mask_without_hand = frame.depth_mask;
	frame.object_rect = cv::Rect(0,0,cols,rows);
	frame.object_mask = frame.depth_mask;
	frame.object_cloud_ptr = frame.cloud_ptr;
}