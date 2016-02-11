#include "pick_pixel.h"

#include <iostream>
using std::cout;
using std::endl;

void PickPixel::mouseCallback(int event, int x, int y, int flags, void* param)
{
	cv::Point2i * point = (cv::Point2i *) param;
	switch(event)
	{
		case CV_EVENT_LBUTTONDOWN:
		{
			point->x = x;
			point->y = y;
			cout << "PickPixel: " << x << " " << y << endl;
		}
		break;
	}
}

PickPixel::PickPixel(std::string window_name)
	: window_name_(window_name),
	pixel_(-1,-1)
{
}

void PickPixel::setMat(const cv::Mat & mat)
{
	cv::namedWindow(window_name_);
	cv::setMouseCallback(window_name_, &PickPixel::mouseCallback, &pixel_);
	cv::imshow(window_name_, mat);
	// could always wait for new selection or something...or loop or I dunno
	// could also just not do this here at all and "wait" for a later waitkey
	//cv::waitKey(1); // ?
}

cv::Point2i PickPixel::getPixel()
{
	return pixel_;
}

void PickPixel::destroyWindows()
{
	cv::destroyWindow(window_name_);

#if 0
	for (std::set<std::string>::iterator i = windows_to_destroy_.begin(); i != windows_to_destroy_.end(); ++i) {
		cv::destroyWindow(*i);
	}
#endif
}

#if 0
bool LearnHistogram::learn(const cv::Mat & image_bgr)
{
	cv::Mat learn_interactive_image = image_bgr.clone();
	const int histSize[] = {params_mask_object_.bins_h, params_mask_object_.bins_s, params_mask_object_.bins_v};

	// need an hsv version of the image...todo: cache this?
	cv::Mat image_hsv;
	cv::cvtColor(image_bgr, image_hsv, CV_BGR2HSV);

	while(true) {
		cv::Rect learn_rect;

		cv::namedWindow("LEARN");
		cv::setMouseCallback("LEARN", &learnWindowMouseCallback, &learn_rect);

		cout << "Select a region and press space" << endl;

		int k = 0;
		while (k != ' ') {
			cv::Mat learn_image = learn_interactive_image.clone();
			cv::rectangle(learn_image, learn_rect, cv::Scalar(255,0,0), 2, 3);
			cv::imshow("LEARN", learn_image);

			k = cv::waitKey(1);
		}

		// proxy for no rect selected
		if (learn_rect.x == 0 && learn_rect.y == 0) return false;

		// set a mask based on the rect
		cv::Mat learn_mask(learn_interactive_image.size(), CV_8UC1, cv::Scalar(0));
		learn_mask(learn_rect).setTo(255); // can supply a mask (like depth mask) as second argument

		showInWindowToDestroy("Learn Mask", learn_mask);

		cv::Mat learn_pixels;
		image_bgr.copyTo(learn_pixels, learn_mask);
		showInWindowToDestroy("Learn Pixels", learn_pixels);

		cv::Mat histogram;
		calcHist( &image_hsv, 1, HistogramConstants::hist_hsv_channels, learn_mask, 
			histogram, 3, histSize, HistogramConstants::hist_hsv_ranges,
			true, // the histogram is uniform
			false );

		// later for sum instead?
		//showMarginalHistograms(histogram);

		//cout << "Press 'y' to add these pixels, 'n' to skip this frame, 'r' to reset to only these pixels, or any other key to try again." << endl;
		cout << "y: add these pixels to histogram" << endl;
		cout << "n: skip this frame without possibility of saving" << endl;
		cout << "i: skip this frame with the possibility of saving" << endl;
		cout << "r: set accumulated histogram to only these pixels" << endl;
		int k_frame = cv::waitKey();
		
		destroyWindows();

		if (k_frame == 'y') {
			if (!histogram_sum_.data) histogram_sum_ = histogram;
			else histogram_sum_ += histogram;
			break;
		}
		else if (k_frame == 'n') {
			return false;
		}
		else if (k_frame == 'i') {
			break;
		}
		else if (k_frame == 'r') {
			histogram_sum_ = histogram;
			break;
		}
	}

	if (histogram_sum_.data) {
		// we now have a new histogram...
		showMarginalHistograms(histogram_sum_);
		// also test with mask_object?
		cout << "If you wish to accept this histogram sum, press 's'." << endl;
		int k_accept = cv::waitKey();
		destroyWindows();
		cv::destroyWindow("LEARN");
		return (k_accept == 's');
	}
	else {
		return false;
	}
}
#endif
