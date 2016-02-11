#include "mask_object.h"

#include "disjoint_set.h"
#include "histogram_util.h"

#include "opencv_utilities.h"

#include <algorithm>
#include <fstream>

#undef min
#undef max

MaskObject::MaskObject(const ParamsMaskObject & params_mask_object)
	: params_mask_object_(params_mask_object),
	seed_search_region_() // this happens anyway
{
	// can always load non-empty files
	//if (params_mask_object_.mask_hand) {
	if (!params_mask_object_.histogram_hand_file.empty()) {
		cv::Mat histogram_hand;
		loadHistogram(params_mask_object_.histogram_hand_file.string(), histogram_hand);
		setHistogramHand(histogram_hand);
	}
	if (!params_mask_object_.histogram_object_file.empty()) {
		cv::Mat histogram_object;
		loadHistogram(params_mask_object_.histogram_object_file.string(), histogram_object);
		setHistogramObject(histogram_object);
	}
}

void MaskObject::setHistogramHand(const cv::Mat & histogram_hand)
{
	histogram_hand_ = histogram_hand;
}

void MaskObject::setHistogramObject(const cv::Mat & histogram_object)
{
	histogram_object_ = histogram_object;
}

cv::Mat MaskObject::getHistogramHand() const
{
	return histogram_hand_.clone(); // clone?
}

cv::Mat MaskObject::getHistogramObject() const
{
	return histogram_object_.clone(); // clone?
}

void MaskObject::resetSeedSearchRegion()
{
	seed_search_region_ = cv::Rect();
}

void MaskObject::save(fs::path const& folder)
{
	{
		// todo: use opencv save/load
		fs::path filename = folder / "seed_search_region.txt";
		std::ofstream file(filename.string().c_str());
		file << seed_search_region_.x << endl;
		file << seed_search_region_.y << endl;
		file << seed_search_region_.width << endl;
		file << seed_search_region_.height << endl;
	}
}

void MaskObject::load(fs::path const& folder)
{
	{
		fs::path filename = folder / "seed_search_region.txt";
		std::ifstream file(filename.string().c_str());
		file >> seed_search_region_.x; 
		file >> seed_search_region_.y; 
		file >> seed_search_region_.width;
		file >> seed_search_region_.height;
	}
}

void MaskObject::getObjectMask(const Frame& frame, cv::Rect & object_rect, cv::Mat & object_mask)
{
	cv::Mat input_object_mask = frame.mat_depth > 0;
	if (params_mask_object_.mask_hand) {
		cv::Mat mask_hand;
		maskHand(frame, mask_hand);

		debug_images_["mask_hand"] = mask_hand;

		cv::Mat mask_not_hand;
		cv::bitwise_not(mask_hand, mask_not_hand);
		cv::bitwise_and(input_object_mask, mask_not_hand, input_object_mask);
	}
	// depth mask is now valid, (non-hand) points

	// initialize seed search region?
	if (seed_search_region_ == cv::Rect()) {
		seed_search_region_.x = 0;
		seed_search_region_.y = 0;
		seed_search_region_.width = frame.mat_depth.cols;
		seed_search_region_.height = frame.mat_depth.rows;

		const float factor = params_mask_object_.initial_seed_rect_scale;
		seed_search_region_.x += seed_search_region_.width * (1 - factor) * 0.5;
		seed_search_region_.y += seed_search_region_.height * (1 - factor) * 0.5;
		seed_search_region_.width *= factor;
		seed_search_region_.height *= factor;
	}

	getObjectMask(frame, input_object_mask, object_rect, object_mask);

	seed_search_region_ = object_rect;
}

void MaskObject::getObjectMask(const Frame & frame, const cv::Mat & input_mask, cv::Rect & object_rect, cv::Mat & object_mask)
{
	cv::Mat input_mask_mutable = input_mask.clone();
	float max_depth_input = -1; // negative means ignore, gets changed as we get more components
	float min_found_depth = 1e6; // big number (lazy peter)

	object_rect = cv::Rect();
	object_mask = cv::Mat::zeros(input_mask.size(), CV_8UC1);
	while(true) {
		cv::Rect this_object_rect;
		cv::Mat this_object_mask;
		float min_depth;
		float max_depth;
		int component_area = getObjectMaskComponent(frame, input_mask_mutable, max_depth_input, this_object_rect, this_object_mask, min_depth, max_depth);
		if (this_object_rect == cv::Rect()) break;
		// is this check necessary?
		if (object_rect == cv::Rect()) object_rect = this_object_rect;
		else object_rect |= this_object_rect;
		cv::bitwise_or(object_mask, this_object_mask, object_mask);
		input_mask_mutable.setTo(0, object_mask);
		max_depth_input = std::max(max_depth_input, max_depth + params_mask_object_.max_disconnected_component_depth_difference);

		// use min_found_depth over all components to constraint max_depth_input as well
		// update: I'm not sure why you would do this...
		// to avoid arms/hands to the side or near back, obviously!
		// TODO: this again
#if 0
		min_found_depth = min(min_found_depth, min_depth);
		max_depth_input = min(max_depth_input, min_found_depth + params.mask_global_max_depth_difference);
#endif

		// update: do debugging some other way
#if 0
		if (debug_if_enabled && params.mask_debug_every_component) {
			showInWindow("mask_debug_every_component", this_object_mask);
			cv::Mat component_color(frame.image_color.size(), frame.image_color.type(), cv::Scalar(0,0,255));
			frame.image_color.copyTo(component_color, this_object_mask);
			showInWindow("mask_debug_every_component (color)", component_color);
			cout << "mask_debug_every_component area: " << component_area << endl;
			cv::waitKey();
		}
#endif
	}

	// could put this check inside the loop as well...
	if (object_rect.width < params_mask_object_.min_object_rect_side || object_rect.height < params_mask_object_.min_object_rect_side) {
		object_rect = cv::Rect();
		object_mask = cv::Mat::zeros(input_mask.size(), CV_8UC1);
	}

	// erode the mask?
	for (int i = 0; i < params_mask_object_.object_mask_erode_iterations; i++) cv::erode(object_mask, object_mask, cv::Mat());
}

int MaskObject::getObjectMaskComponent(const Frame & frame, const cv::Mat & input_mask, float input_max_depth, cv::Rect & object_rect, cv::Mat & object_mask, float & min_depth, float & max_depth)
{
	// find the closest point within seedSearchRegion
	cv::Mat depth_image_roi = frame.mat_depth(seed_search_region_);
	double minVal;
	cv::Point minLoc;
	cv::Mat mask_roi = input_mask(seed_search_region_);

	if (cv::countNonZero(mask_roi) == 0) {
		object_rect = cv::Rect();
		object_mask = cv::Mat();
		return 0;
	}

	cv::minMaxLoc(depth_image_roi, &minVal, NULL, &minLoc, NULL, mask_roi);
	cv::Point seedPoint = seed_search_region_.tl() + minLoc;

	if (input_max_depth > 0 && minVal > input_max_depth) {
		object_rect = cv::Rect();
		object_mask = cv::Mat();
		return 0;
	}

	// floodFill for depth masking
	cv::Mat depthForFloodFill = cv::Mat::zeros(frame.mat_depth.size(), frame.mat_depth.type()); // later assuming CV_32FC1
	frame.mat_depth.copyTo(depthForFloodFill, input_mask);
	cv::Mat floodFillMask = cv::Mat::zeros(frame.mat_depth.size() + cv::Size(2,2), CV_8UC1);
	cv::Rect floodRect;
	int connectivity = 8;
	int newMaskVal = 255;
	int flags = connectivity + (newMaskVal << 8) + cv::FLOODFILL_MASK_ONLY;
	int area = cv::floodFill(depthForFloodFill, floodFillMask, seedPoint, 0, &floodRect, cv::Scalar(params_mask_object_.max_connected_component_depth_difference), cv::Scalar(params_mask_object_.max_connected_component_depth_difference), flags);

	object_rect = floodRect;
	object_mask = floodFillMask (cv::Rect(1,1,frame.mat_depth.cols,frame.mat_depth.rows)); // maybe clone to get rid of extra data?

	double min_depth_double;
	double max_depth_double;
	cv::minMaxLoc(frame.mat_depth, &min_depth_double, &max_depth_double, NULL, NULL, object_mask);
	min_depth = min_depth_double;
	max_depth = max_depth_double;

	return area;
}


void MaskObject::maskHand(const Frame & frame, cv::Mat & mask_hand)
{
	if (histogram_hand_.dims != 3 || histogram_object_.dims != 3) {
		throw std::runtime_error("missing correct histograms");
	}

	// get an hsv here?
	cv::Mat mat_color_bgr;
	cv::cvtColor(frame.mat_color_bgra, mat_color_bgr, CV_BGRA2BGR);
	cv::Mat mat_color_hsv;
	cv::cvtColor(mat_color_bgr, mat_color_hsv, CV_BGR2HSV);

	mask_hand = maskHandWithHistograms(mat_color_hsv);

	// remove for speed:
	std::vector<cv::Mat> debug_images;

	debug_images.push_back(mask_hand.clone());

	mask_hand = removeSmallComponentsWithDisjointSet(mask_hand, params_mask_object_.mask_hand_min_component_area_before_morphology);

	debug_images.push_back(mask_hand.clone());

	// erode?
	for (int i = 0; i < params_mask_object_.mask_hand_erode_iterations; i++) cv::erode(mask_hand, mask_hand, cv::Mat());

	// dilate?
	for (int i = 0; i < params_mask_object_.mask_hand_dilate_iterations; i++) cv::dilate(mask_hand, mask_hand, cv::Mat());

	debug_images.push_back(mask_hand.clone());

	mask_hand = removeSmallComponentsWithDisjointSet(mask_hand, params_mask_object_.mask_hand_min_component_area_after_morphology);

	debug_images.push_back(mask_hand.clone());

	// this is false in the old code...is it still interesting?
	if (params_mask_object_.mask_hand_floodfill) {
		mask_hand = floodFillExpandMask(frame.mat_color_bgra, mask_hand);
	}

	cv::Mat all_debug_images = createMxN(2,2,debug_images);
	debug_images_["mask_hand_all"] = all_debug_images;
}

cv::Mat MaskObject::maskHandWithHistograms(const cv::Mat & mat_color_hsv)
{
	cv::Mat back_project_hand;
	cv::calcBackProject(&mat_color_hsv, 1, HistogramConstants::hist_hsv_channels, histogram_hand_, back_project_hand, HistogramConstants::hist_hsv_ranges, 1.0);
	cv::Mat back_project_object;
	cv::calcBackProject(&mat_color_hsv, 1, HistogramConstants::hist_hsv_channels, histogram_object_, back_project_object, HistogramConstants::hist_hsv_ranges, 1.0);

	// do we really need floats here?
	back_project_hand.convertTo(back_project_hand, CV_32F, 1./255);
	back_project_object.convertTo(back_project_object, CV_32F, 1./255);

	cv::Mat back_project_sum = back_project_hand + back_project_object;

	cv::Mat back_project_relative = back_project_hand / back_project_object;

	cv::Mat back_project_mask_float;
	cv::threshold(back_project_relative, back_project_mask_float, params_mask_object_.mask_hand_backproject_threshold, 255, CV_THRESH_BINARY);

	cv::Mat back_project_mask_char;
	back_project_mask_float.convertTo(back_project_mask_char, CV_8U);

#if 0
	debug_images_["back_project_hand"] = back_project_hand;
	debug_images_["back_project_object"] = back_project_object;
	debug_images_["back_project_relative"] = back_project_relative;
	debug_images_["back_project_mask_float"] = back_project_mask_float;
#endif
	std::vector<cv::Mat> image_v;
	image_v.push_back(floatC1toCharC3(back_project_hand));
	image_v.push_back(floatC1toCharC3(back_project_object));
	image_v.push_back(floatC1toCharC3(back_project_relative));
	image_v.push_back(floatC1toCharC3(back_project_mask_float));
	cv::Mat back_project_four = createMxN(2, 2, image_v);
	debug_images_["back_project_four"] = back_project_four;

	return back_project_mask_char;
}


cv::Mat MaskObject::removeSmallComponentsWithDisjointSet(const cv::Mat& input_mask, int min_component_area)
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


cv::Mat MaskObject::floodFillExpandMask(const cv::Mat & image_color, const cv::Mat& input_mask)
{
	// set up accumulatedFloodMask
	const static int connectivity = 8;
	const static int hand_mask_value = 255;
	const static int non_object_mask_value = 100; // not used?
	cv::Mat accumulatedFloodMask(input_mask.size() + cv::Size(2,2), CV_8UC1, cv::Scalar(0));
	cv::Mat accumulatedFloodMaskImageSize = accumulatedFloodMask(cv::Rect(1,1,input_mask.cols,input_mask.rows));

	cv::Mat imageForFloodFill = image_color.clone(); // clone necessary?
	const cv::Scalar lowDiff(params_mask_object_.mask_floodfill_expand_diff, params_mask_object_.mask_floodfill_expand_diff, params_mask_object_.mask_floodfill_expand_diff);
	const cv::Scalar highDiff(params_mask_object_.mask_floodfill_expand_diff, params_mask_object_.mask_floodfill_expand_diff, params_mask_object_.mask_floodfill_expand_diff);
	const static int flags = connectivity + (hand_mask_value << 8) + cv::FLOODFILL_MASK_ONLY;
	int area_sum = 0;
	for (int row = 0; row < imageForFloodFill.rows; row++) {
		for (int col = 0; col < imageForFloodFill.cols; col++) {
			if (!input_mask.at<unsigned char>(row, col)) continue;
			int area = cv::floodFill(imageForFloodFill, accumulatedFloodMask, cv::Point(col, row), 0, NULL, lowDiff, highDiff, flags);
			area_sum += area;
		}
	}

	cv::Mat flood_fill_mask_result; // result
	cv::threshold(accumulatedFloodMaskImageSize, flood_fill_mask_result, non_object_mask_value+1, 255, CV_THRESH_BINARY);

	return flood_fill_mask_result;
}


std::map<std::string, cv::Mat> const& MaskObject::getDebugImages() const
{
	return debug_images_;
}
