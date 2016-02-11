#pragma once

#include "typedefs.h"
#include "parameters.h"
#include "ObjectModeler.h"

#include <OpenNI.h>

class OpenNIInput
{
public:
	OpenNIInput(Parameters& params);

	void run();

	void cloud_callback(CloudT::ConstPtr cloud);

protected:
	ObjectModeler object_modeler;

	openni::Device device;
	openni::VideoStream depth, color;

#if 0
	boost::mutex mtx;
	boost::condition_variable cond;
	CloudT::ConstPtr last_cloud_ptr;
#endif

	
};

