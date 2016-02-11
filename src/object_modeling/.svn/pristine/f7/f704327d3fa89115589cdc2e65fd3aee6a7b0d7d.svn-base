#include "frame_provider_openni2.h"

#include <boost/scoped_ptr.hpp>

int main(int argc, char* argv[])
{
#ifdef _WIN32
#if 1
	// windows: buffer stdout better!
	const int console_buffer_size = 4096;
	char buf[console_buffer_size];
	setvbuf(stdout, buf, _IOLBF, console_buffer_size);
#else
	cout << "WARNING: SLOW COUT" << endl;
#endif
#endif

	enum WindowMode {
		WINDOW_MODE_MASK,
		WINDOW_MODE_BOTH
	};
	WindowMode window_mode = WINDOW_MODE_MASK;

	bool openni_auto = false;
	bool recording = false;
	FrameProviderOpenni2Params params;
	params.auto_exposure = openni_auto;
	params.auto_white_balance = openni_auto;

	boost::scoped_ptr<FrameProviderOpenni2> frame_provider_openni2;
	frame_provider_openni2.reset(new FrameProviderOpenni2(params));

	while(true) {
		cv::Mat mat_color;
		cv::Mat mat_depth;
		bool frame_valid = frame_provider_openni2->getNextFrame(mat_color, mat_depth);
		if (!frame_valid) break;

		//////// show image
		if (window_mode == WINDOW_MODE_MASK) {
			// wasteful?
			cv::destroyWindow("color");
			cv::destroyWindow("depth");

			// just needed for masking?
			cv::Mat depth_8;
			mat_depth.convertTo(depth_8, CV_8U, 255./10.); // depth is in meters, so this makes 10 meters the max value...

			cv::Mat masked_color;
			mat_color.copyTo(masked_color, depth_8);

			cv::imshow("masked color", masked_color);
		}
		else if (window_mode == WINDOW_MODE_BOTH) {
			// wasteful?
			cv::destroyWindow("masked color");

			cv::imshow("color", mat_color);
			cv::imshow("depth", mat_depth);
		}
		else {
			cout << "UNKNOWN WINDOW MODE!" << endl;
		}


		int key = cv::waitKey(1);
		if (key == 'a') {
			openni_auto = !openni_auto;
			frame_provider_openni2->setAutoExposure(openni_auto);
			frame_provider_openni2->setAutoWhiteBalance(openni_auto);
		}
		else if (key == 's') {
			recording = !recording;
			frame_provider_openni2->setRecording(recording);
		}
		else if (key == 'w') {
			//nextWindowMode();
		}
		else if (key == 'q') {
			break; // ?
		}

	}



	return 0;
}