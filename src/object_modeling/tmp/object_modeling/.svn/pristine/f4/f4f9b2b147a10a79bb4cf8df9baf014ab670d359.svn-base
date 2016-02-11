#include "stdafx.h"
#include "OpenNIInput.h"
using std::cout;
using std::endl;

OpenNIInput::OpenNIInput(Parameters& params)
	: object_modeler(params)
{
	openni::Status rc = openni::OpenNI::initialize();
	if (rc != openni::STATUS_OK)
	{
		printf("Initialize failed\n%s\n", openni::OpenNI::getExtendedError());
		exit(1);
	}

	rc = device.open(openni::ANY_DEVICE);
	if (rc != openni::STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		exit(1);
	}

	// inject some settings
	device.setDepthColorSyncEnabled(true);
	device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);


	rc = depth.create(device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = depth.start();
		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			depth.destroy();
		}
	}
	else
	{
		printf("Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = color.start();
		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	if (!depth.isValid() || !color.isValid())
	{
		printf("No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		exit(1);
	}

	// The fun camera settings
	openni::CameraSettings* color_settings = color.getCameraSettings();
	if (!color_settings) {
		printf("No color_settings");
		exit(1);
	}
	color_settings->setAutoExposureEnabled(false);
	color_settings->setAutoWhiteBalanceEnabled(false);

	// set a good resolution
	const openni::SensorInfo& color_sensor_info = color.getSensorInfo();
	const openni::SensorInfo& depth_sensor_info = depth.getSensorInfo();
	const openni::Array<openni::VideoMode>& color_modes = color_sensor_info.getSupportedVideoModes();
	const openni::Array<openni::VideoMode>& depth_modes = depth_sensor_info.getSupportedVideoModes();
	const openni::VideoMode* good_color_mode = NULL;
	for (int i = 0; i < color_modes.getSize(); ++i) {
		if (color_modes[i].getFps() == 30 && 
			color_modes[i].getResolutionX() == 640 && 
			color_modes[i].getResolutionY() == 480 &&
			color_modes[i].getPixelFormat() == openni::PIXEL_FORMAT_RGB888 ) {
				good_color_mode = &color_modes[i];
				break;
		}
	}
	if (good_color_mode) {
		color.setVideoMode(*good_color_mode);
	}
	else {
		printf("Couldn't find good_color_mode");
		exit(1);
	}

	const openni::VideoMode* good_depth_mode = NULL;
	for (int i = 0; i < depth_modes.getSize(); ++i) {
		if (depth_modes[i].getFps() == 30 && 
			depth_modes[i].getResolutionX() == 640 && 
			depth_modes[i].getResolutionY() == 480 &&
			depth_modes[i].getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_100_UM ) {
				good_depth_mode = &depth_modes[i];
				break;
		}
	}
	if (good_depth_mode) {
		depth.setVideoMode(*good_depth_mode);
	}
	else {
		printf("Couldn't find good_depth_mode");
		exit(1);
	}

#if 0
	openni::VideoMode depthVideoMode = depth.getVideoMode();
	openni::VideoMode colorVideoMode = color.getVideoMode();

	int depthWidth = depthVideoMode.getResolutionX();
	int depthHeight = depthVideoMode.getResolutionY();
	int colorWidth = colorVideoMode.getResolutionX();
	int colorHeight = colorVideoMode.getResolutionY();

	if (depthWidth == colorWidth &&
		depthHeight == colorHeight)
	{
		m_width = depthWidth;
		m_height = depthHeight;
	}
	else
	{
		printf("Error - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
			depthWidth, depthHeight,
			colorWidth, colorHeight);
		return openni::STATUS_ERROR;
	}
#endif

}

void OpenNIInput::run()
{

#if 0
	pcl::Grabber* grabber = NULL;
	while(true) {
		try {
			grabber = new pcl::OpenNIGrabber();
		}
		catch (pcl::PCLIOException e) {
			cout << "Waiting for OpenNI device..." << endl;
			continue;
		}
		break;
	}

	boost::function<void (const CloudT::ConstPtr&)> f = boost::bind (&OpenNIInput::cloud_callback, this, _1);

	grabber->registerCallback (f);
	grabber->start ();

	while(!object_modeler.wasStopped()) {
		FrameT frame;
		{
			boost::mutex::scoped_lock lock(mtx);
			cond.wait(lock);
			frame.cloud_ptr.reset(new CloudT);
			*frame.cloud_ptr = *last_cloud_ptr;
		}
		object_modeler.processFrame(frame);
	}
	
	grabber->stop ();
#endif
}

void OpenNIInput::cloud_callback(CloudT::ConstPtr cloud)
{

#if 0
	if (!object_modeler.wasStopped()) {
		boost::mutex::scoped_lock lock(mtx);
		last_cloud_ptr = cloud;
		cond.notify_one();
	}
#endif
}
