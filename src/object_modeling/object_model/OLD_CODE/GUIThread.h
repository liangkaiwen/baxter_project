#pragma once

#include "Semaphore.h"

class GUIThread
{
public:
	GUIThread(void);

	void enqueue(std::string window_name, cv::Mat image);
	void enqueueCallback(std::string window_name, cv::MouseCallback cb, void* param);
	void enqueueDestroy(std::string window_name);
	int enqueueWaitKeyRequest(bool hold);
	void stop();
	void operator() ();

protected:
	boost::mutex mtx;
	std::list<std::pair<std::string, cv::Mat> > image_queue;
	std::list<boost::tuple<std::string, cv::MouseCallback, void*> > callback_queue;
	std::list<std::string> destroy_queue;
	bool do_stop;

	struct WaitKeyRequest {
		bool hold;
		int result;
		boost::condition_variable cond;
	};
	std::list<WaitKeyRequest*> waitkey_request_queue;
	//boost::mutex waitkey_request_mutex;
};

