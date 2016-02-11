#include <stdio.h>

#include <iostream>
using std::cout;
using std::endl;

#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/scoped_ptr.hpp>


// opencv utilities
#include "opencv_utilities.h"

// frame provider
#ifdef FRAME_PROVIDER_PCL
#include "frame_provider_pcd.h"
#endif



//////////////////////////////

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


	/////////////
	// boost program options
	fs::path input_pcd;
	fs::path output_png;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input_pcd", po::value<fs::path>(&input_pcd), "input_pcd")
		("output_png", po::value<fs::path>(&output_png), "output_png")
	
		;
	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (std::exception & e) {
		cout << desc << endl;
		cout << e.what() << endl;
		return 1;
	}
	if (vm.count("help")) {
		cout << "desc" << endl;
		return 1;
	}

	if (input_pcd.empty() || output_png.empty()) {
		cout << "desc" << endl;
		return 1;
	}

	if (!fs::exists(output_png) && !fs::create_directories(output_png)) {
		cout << "couldn't create: " << output_png << endl;
		return 1;
	}

	boost::scoped_ptr<FrameProviderBase> frame_provider;
	//boost::shared_ptr<FrameProviderPCD> core_frame_provider (new FrameProviderPCD(input_pcd));
	//frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
	frame_provider.reset(new FrameProviderPCD(input_pcd));

	cv::Mat mat_color, mat_depth;
	int frame_counter = 0;
	float depth_factor = 10000;
	while (frame_provider->getNextFrame(mat_color, mat_depth)) {
		// do stuff
		// format: #####-depth.png (with 100um units)
		//cv::Mat depth_png = cv::imread(expected_depth_file.string(), CV_LOAD_IMAGE_ANYDEPTH);
		cv::Mat depth_png;
		mat_depth.convertTo(depth_png, CV_16U, depth_factor);

		fs::path file_color = output_png / (boost::format("%05d-color.png") % frame_counter).str();
		fs::path file_depth = output_png / (boost::format("%05d-depth.png") % frame_counter).str();

		cv::imwrite(file_color.string(), mat_color);
		cv::imwrite(file_depth.string(), depth_png);
		
		frame_counter++;
	}


	return 0;
}
