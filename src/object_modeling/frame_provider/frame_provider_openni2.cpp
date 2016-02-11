#include "frame_provider_openni2.h"

#include <boost/timer.hpp>

FrameProviderOpenni2::FrameProviderOpenni2(FrameProviderOpenni2Params & params)
    : FrameProviderBase(),
	params(params),
	frame_count(0),
	frame_counter(0)
{
	// init openni
	openni::Status rc = openni::OpenNI::initialize();
	if (rc != openni::STATUS_OK)
	{
		printf("OpenNI2 Initialize failed\n%s\n", openni::OpenNI::getExtendedError());
		throw std::runtime_error("FrameProviderOpenni2 constructor");
	}

	init();
}

FrameProviderOpenni2::~FrameProviderOpenni2()
{
	setRecording(false);
	openni::OpenNI::shutdown();
}

void FrameProviderOpenni2::init()
{
	openni::Status rc;
	if (params.file.empty()) {
		rc = device.open(openni::ANY_DEVICE);
	}
	else {
		rc = device.open(params.file.string().c_str());
	}
	if (rc != openni::STATUS_OK)
	{
		printf("OpenNI2 Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		throw std::runtime_error("FrameProviderOpenni2 constructor");
	}

	rc = depth_stream.create(device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = depth_stream.start();
		if (rc != openni::STATUS_OK)
		{
			printf("OpenNI2 Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			depth_stream.destroy();
		}
	}
	else
	{
		printf("OpenNI2 Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	rc = color_stream.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = color_stream.start();
		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color_stream.destroy();
		}
	}
	else
	{
		printf("Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	if (!depth_stream.isValid() || !color_stream.isValid())
	{
		printf("No valid streams.\n");
		openni::OpenNI::shutdown();
		throw std::runtime_error("FrameProviderOpenni2 constructor");
	}

	if (!params.file.empty()) {
		int depth_frame_count = device.getPlaybackControl()->getNumberOfFrames(depth_stream);
		int color_frame_count = device.getPlaybackControl()->getNumberOfFrames(color_stream);
		printf("depth_frame_count: %d\n", depth_frame_count);
		printf("color_frame_count: %d\n", color_frame_count);
		if (depth_frame_count != color_frame_count) {
			cout << "Frame count mismatch (not supported)" << endl;
			throw std::runtime_error("Frame count mismatch");
		}
		frame_count = depth_frame_count;
		frame_counter = 0;
	}

	///////////////
	// set a good resolution
	// This could well be generalized
	const openni::SensorInfo& color_sensor_info = color_stream.getSensorInfo();
	const openni::SensorInfo& depth_sensor_info = depth_stream.getSensorInfo();
	const openni::Array<openni::VideoMode>& color_modes = color_sensor_info.getSupportedVideoModes();
	const openni::Array<openni::VideoMode>& depth_modes = depth_sensor_info.getSupportedVideoModes();
	const openni::VideoMode* good_color_mode = NULL;
	for (int i = 0; i < color_modes.getSize(); ++i) {
		if (color_modes[i].getFps() == params.fps && 
			color_modes[i].getResolutionX() == params.resolution_x && 
			color_modes[i].getResolutionY() == params.resolution_y &&
			color_modes[i].getPixelFormat() == openni::PIXEL_FORMAT_RGB888) {
				good_color_mode = &color_modes[i];
				break;
		}
	}
	if (good_color_mode) {
		color_stream.setVideoMode(*good_color_mode);
	}
	else {
		printf("Couldn't find good_color_mode\n");
		throw std::runtime_error("FrameProviderOpenni2 constructor");
	}

	const openni::VideoMode* good_depth_mode = NULL;
	for (int i = 0; i < depth_modes.getSize(); ++i) {
		if (depth_modes[i].getFps() == params.fps && 
			depth_modes[i].getResolutionX() == params.resolution_x && 
			depth_modes[i].getResolutionY() == params.resolution_y &&
			depth_modes[i].getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_100_UM ) {
				good_depth_mode = &depth_modes[i];
				break;
		}
	}
	if (good_depth_mode) {
		depth_stream.setVideoMode(*good_depth_mode);
	}
	else {
		printf("Couldn't find good_depth_mode\n");
		throw std::runtime_error("FrameProviderOpenni2 constructor");
	}

	// set mirroring off
	color_stream.setMirroringEnabled(false);
	depth_stream.setMirroringEnabled(false);

	// no cropping (yeah, right)
	color_stream.resetCropping();
	depth_stream.resetCropping();

	if (params.file.empty()) {
		// the all important OpenNI 2.0 settings (for live input)
		rc = device.setDepthColorSyncEnabled(true);
		if (rc != openni::STATUS_OK) {
			printf("FAIL: device.setDepthColorSyncEnabled(true);\n");
			throw std::runtime_error("FrameProviderOpenni2 constructor");
		}
		rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
		if (rc != openni::STATUS_OK) {
			printf("FAIL: device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);\n");
			throw std::runtime_error("FrameProviderOpenni2 constructor");
		}

		setAutoExposure(params.auto_exposure);
		setAutoWhiteBalance(params.auto_white_balance);

	}
	else {
		device.getPlaybackControl()->setSpeed(-1);
		device.getPlaybackControl()->setRepeatEnabled(false);
	}
}

void FrameProviderOpenni2::setAutoExposure(bool value)
{
	if (params.file.empty()) {
		openni::CameraSettings* color_settings = color_stream.getCameraSettings();
		openni::Status rc = color_settings->setAutoExposureEnabled(value);
		if (rc != openni::STATUS_OK) {
			cout << "Warning: setAutoExposure not STATUS_OK" << endl;
		}
	}
}
	
void FrameProviderOpenni2::setAutoWhiteBalance(bool value)
{
	if (params.file.empty()) {
		openni::CameraSettings* color_settings = color_stream.getCameraSettings();
		openni::Status rc = color_settings->setAutoWhiteBalanceEnabled(value);
		if (rc != openni::STATUS_OK) {
			cout << "Warning: setAutoWhiteBalance not STATUS_OK" << endl;
		}
	}
}

bool FrameProviderOpenni2::getNextFrameRefs(openni::VideoFrameRef & color_frame_ref, openni::VideoFrameRef & depth_frame_ref)
{
	const static bool verbose = false;

	if (!color_stream.isValid() || !depth_stream.isValid()) {
		if (verbose) cout << "false for invalid stream" << endl;
		return false;
	}

	if (frame_count > 0 && frame_counter++ >= frame_count) {
		if (verbose) cout << "false for frame_counter: " << frame_counter << endl;
		return false;
	}

	openni::Status rc = openni::STATUS_OK;
	openni::VideoStream* streams[] = {&color_stream, &depth_stream};
	int changedIndex = -1;
	bool have_depth = false;
	bool have_color = false;
	while (rc == openni::STATUS_OK)
	{
		boost::timer t;
		if (verbose) cout << "About to waitForAnyStream" << endl;
		const static int ms_timeout = 1000;
		rc = openni::OpenNI::waitForAnyStream(streams, 2, &changedIndex, ms_timeout);
		if (verbose) cout << "Time in waitForAnyStream: " << t.elapsed() << endl;
		if (rc == openni::STATUS_OK)
		{
			switch (changedIndex)
			{
			case 0:
				color_stream.readFrame(&color_frame_ref);
				if (verbose) cout << "Got color frame: " << color_frame_ref.getFrameIndex() << endl;
				have_color = true;
				break;
			case 1:
				depth_stream.readFrame(&depth_frame_ref);
				if (verbose) cout << "Got depth frame: " << depth_frame_ref.getFrameIndex() << endl;
				have_depth = true;
				break;
			default:
				throw std::runtime_error("Unexpected Stream");
				return false;
			}
		}
		else {
			cout << "Warning: problem with waitForAnyStream" << endl;
			cout << "OpenNI::getExtendedError(): " << openni::OpenNI::getExtendedError() << endl;
			rc =  openni::STATUS_OK; // so we can loop
			continue;
		}

		// see if we have both and they are within a timestamp diff of each other
		if (have_color && have_depth) {
			const int max_microseconds = 10 * 1000; // 10 ms...very generous
			uint64_t color_stamp = color_frame_ref.getTimestamp();
			uint64_t depth_stamp = depth_frame_ref.getTimestamp();
			int abs_timestamp_diff = depth_stamp > color_stamp ? depth_stamp - color_stamp : color_stamp - depth_stamp;
			if ( abs_timestamp_diff < max_microseconds ) {
				return true;
			}
			else {
				if (verbose) cout << "Timestamp diff: " << abs_timestamp_diff << endl;
			}
		}
	}

	return false;
}

bool FrameProviderOpenni2::getNextFrame(cv::Mat & color, cv::Mat & depth)
{
	openni::VideoFrameRef color_frame_ref;
	openni::VideoFrameRef depth_frame_ref;
	bool got_frame = getNextFrameRefs(color_frame_ref, depth_frame_ref);
	if (!got_frame) return false;

	// get the data as cv mats
	cv::Mat color_mat(color_frame_ref.getHeight(), color_frame_ref.getWidth(), CV_8UC3);
	cv::Mat depth_mat(depth_frame_ref.getHeight(), depth_frame_ref.getWidth(), CV_16UC1);
	// just copy memory :)
	if (color_mat.rows * color_mat.cols * color_mat.elemSize() != color_frame_ref.getDataSize()) {
		printf("bad color size"); exit(1);
	}
	if (depth_mat.rows * depth_mat.cols * depth_mat.elemSize() != depth_frame_ref.getDataSize()) {
		printf("bad depth size"); exit(1);
	}
	memcpy(color_mat.data, color_frame_ref.getData(), color_frame_ref.getDataSize());
	memcpy(depth_mat.data, depth_frame_ref.getData(), depth_frame_ref.getDataSize());

	// rgb to bgra
	cv::cvtColor(color_mat, color_mat, CV_RGB2BGRA);

	// depth to meters
	cv::Mat depth_mat_float;
	float depth_to_meters = 1.f/10000.f;
	depth_mat.convertTo(depth_mat_float, CV_32F, depth_to_meters);

	// set up frame
	color = color_mat;
	depth = depth_mat_float;

	return true;
}

void FrameProviderOpenni2::skipNextFrame()
{
	openni::VideoFrameRef color_frame_ref;
	openni::VideoFrameRef depth_frame_ref;
    bool got_frame_ignore = getNextFrameRefs(color_frame_ref, depth_frame_ref);
}

void FrameProviderOpenni2::reset()
{
	// this works in the middle, but not at the end:
	/*
	bool device_valid = device.getPlaybackControl()->isValid();
	cout << "reset device_valid: " << device_valid << endl;
	bool streams_valid = color_stream.isValid() && depth_stream.isValid();
	cout << "reset streams_valid: " << streams_valid << endl;
	device.getPlaybackControl()->seek(color_stream, 0);
	frame_counter = 0;
	*/
	device.close();
	init();
}

std::string FrameProviderOpenni2::getFreshFilename()
{
	char buffer[1024];
	std::string prefix_could_be_param = "oni_recording";
	sprintf(buffer, "%s_%d.oni", prefix_could_be_param.c_str(), time(0));
	return buffer;
}

void FrameProviderOpenni2::setRecording(bool value)
{
	// could be params
	const static bool allow_lossy_compress_color = true;
	const static bool allow_lossy_compress_depth = false;

	// null pointer means not recording	
	if (value) {
		if (recorder_ptr) return;
		// create a new recorder
		std::string filename = getFreshFilename();
		recorder_ptr.reset(new openni::Recorder);
		recorder_ptr->create(filename.c_str());
		recorder_ptr->attach(depth_stream, allow_lossy_compress_depth);
		recorder_ptr->attach(color_stream, allow_lossy_compress_color);
		recorder_ptr->start();
		printf("recorder started...writing to %s\n", filename.c_str());
	}
	else {
		if (!recorder_ptr) return;
		recorder_ptr->stop();
		recorder_ptr->destroy();
		recorder_ptr.reset();
		printf("recorder stopped.");
	}
}
