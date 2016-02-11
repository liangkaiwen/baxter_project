#include "stdafx.h"

#include "LearnHistogram.h"

#include "histogramUtil.h"
#include "opencvUtil.h"

using std::cout;
using std::endl;

void learnWindowMouseCallback(int event, int x, int y, int flags, void* param)
{
	cv::Rect *rect = (cv::Rect *)param; 
	switch(event)
	{
	case CV_EVENT_LBUTTONDOWN:
		{
			rect->x = 0;
			rect->y = 0;
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		{
			if(flags & CV_EVENT_FLAG_LBUTTON)
			{
				if((rect->x == 0 && rect->y == 0))
				{
					rect->x = x;
					rect->y = y;
				}
				else
				{
					rect->width = std::max(x - rect->x, 1);
					rect->height = std::max(y - rect->y, 1);
				}
			}
		}
		break;

	default:
		break;
	}
}

LearnHistogram::LearnHistogram(const Parameters& params)
	: params(params)
{
}

void LearnHistogram::showInWindowToDestroy(const std::string& name, const cv::Mat& image)
{
	windowsToDestroy.insert(name);
	showInWindow(name, image);
}

void LearnHistogram::destroyWindows()
{
	for (std::set<std::string>::iterator i = windowsToDestroy.begin(); i != windowsToDestroy.end(); ++i) {
		cv::destroyWindow(*i);
	}
}

void LearnHistogram::showMarginalHistograms(const cv::Mat& histogram)
{
	showInWindowToDestroy("H Histogram", view3DHistogram1D(histogram, 0));
	showInWindowToDestroy("S Histogram", view3DHistogram1D(histogram, 1));
	showInWindowToDestroy("V Histogram", view3DHistogram1D(histogram, 2));
	showInWindowToDestroy("S-V Histogram", view3DHistogram2D(histogram, 0));
	showInWindowToDestroy("H-V Histogram", view3DHistogram2D(histogram, 1));
	showInWindowToDestroy("H-S Histogram", view3DHistogram2D(histogram, 2));
}

bool LearnHistogram::learn(const FrameT& frame)
{
	cv::Mat learn_interactive_image = frame.image_color.clone();
	const int histSize[] = {params.mask_hand_learn_hbins, params.mask_hand_learn_sbins, params.mask_hand_learn_vbins};

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
		frame.image_color.copyTo(learn_pixels, learn_mask);
		showInWindowToDestroy("Learn Pixels", learn_pixels);

		cv::Mat histogram;
		calcHist( &frame.image_color_hsv, 1, HistogramConstants::hist_hsv_channels, learn_mask, 
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
			if (!histogram_sum.data) histogram_sum = histogram;
			else histogram_sum += histogram;
			break;
		}
		else if (k_frame == 'n') {
			return false;
		}
		else if (k_frame == 'i') {
			break;
		}
		else if (k_frame == 'r') {
			histogram_sum = histogram;
			break;
		}
	}

	if (histogram_sum.data) {
		// we now have a new histogram...
		showMarginalHistograms(histogram_sum);
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

void LearnHistogram::reset()
{
	histogram_sum = cv::Mat();
}
	
void LearnHistogram::init(cv::Mat initial_histogram)
{
	histogram_sum = initial_histogram;
}

cv::Mat LearnHistogram::getHistogram()
{
	return histogram_sum;
}