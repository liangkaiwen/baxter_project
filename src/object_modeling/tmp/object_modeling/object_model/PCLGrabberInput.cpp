#include "stdafx.h"
#include "PCLGrabberInput.h"
using std::cout;
using std::endl;


PCLGrabberInput::PCLGrabberInput(Parameters& params)
	: object_modeler(params)
{

}

void PCLGrabberInput::run()
{
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

	boost::function<void (const CloudT::ConstPtr&)> f = boost::bind (&PCLGrabberInput::cloud_callback, this, _1);

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
}

void PCLGrabberInput::cloud_callback(CloudT::ConstPtr cloud)
{
	if (!object_modeler.wasStopped()) {
		boost::mutex::scoped_lock lock(mtx);
		last_cloud_ptr = cloud;
		cond.notify_one();
	}
}
