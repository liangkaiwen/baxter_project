#pragma once

#include "typedefs.h"
#include "parameters.h"
#include "ObjectModeler.h"


#include <pcl/io/openni_grabber.h>

class PCLGrabberInput
{
public:
	PCLGrabberInput(Parameters& params);

	void run();

	void cloud_callback(CloudT::ConstPtr cloud);

protected:
	boost::mutex mtx;
	boost::condition_variable cond;
	CloudT::ConstPtr last_cloud_ptr;

	ObjectModeler object_modeler;
};

