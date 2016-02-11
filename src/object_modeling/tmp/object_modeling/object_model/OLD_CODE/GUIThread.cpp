#include "StdAfx.h"
#include "GUIThread.h"

GUIThread::GUIThread(void)
	: do_stop(false)
{
}

void GUIThread::stop()
{
	boost::mutex::scoped_lock lock(mtx);
	do_stop = true;
}

int GUIThread::enqueueWaitKeyRequest(bool hold)
{
	WaitKeyRequest req;
	req.hold = hold;
	boost::mutex::scoped_lock lock(mtx);
	waitkey_request_queue.push_back(&req);
	req.cond.wait(lock);
	return req.result;
}

void GUIThread::enqueue(std::string window_name, cv::Mat image)
{
	boost::mutex::scoped_lock lock(mtx);
	image_queue.push_back(std::make_pair(window_name, image));
}

void GUIThread::enqueueCallback(std::string window_name, cv::MouseCallback cb, void* param)
{
	boost::mutex::scoped_lock lock(mtx);
	callback_queue.push_back(boost::make_tuple(window_name, cb, param));
}

void GUIThread::enqueueDestroy(std::string window_name)
{
	boost::mutex::scoped_lock lock(mtx);
	destroy_queue.push_back(window_name);
}

void GUIThread::operator() ()
{
	cout << "GUIThread running" << endl;

	while(!do_stop) {
		// lock the show images queue, and keep only the most recent
		mtx.lock();
		std::map<std::string, cv::Mat> most_recent_map;
		while(!image_queue.empty()) {
			most_recent_map.insert(image_queue.front());
			image_queue.pop_front();
		}
		mtx.unlock();

		// show these images
		for (std::map<std::string, cv::Mat>::iterator iter = most_recent_map.begin(); iter != most_recent_map.end(); ++iter) {
			cv::namedWindow(iter->first);
			cv::imshow(iter->first, iter->second);
		}

		// now lock again and do the callback and destruction within the lock
		mtx.lock();
		while (!callback_queue.empty()) {
			cv::setMouseCallback(callback_queue.front().get<0>(), callback_queue.front().get<1>(), callback_queue.front().get<2>());
			callback_queue.pop_front();
		}

		while (!destroy_queue.empty()) {
			cv::destroyWindow(destroy_queue.front());
			destroy_queue.pop_front();
		}
		mtx.unlock();

		int default_result = cv::waitKey(50);
		//cout << "default_result: " << default_result << endl;
		{
			boost::mutex::scoped_lock lock(mtx);
			if (!waitkey_request_queue.empty()) {
				WaitKeyRequest *local_req = waitkey_request_queue.front();
				waitkey_request_queue.pop_front();
				if (local_req->hold) {
					cout << "waiting for HOLD result" << endl;
					local_req->result = cv::waitKey(0);
					cout << "setting HOLD result to: " << local_req->result << endl;
				}
				else {
					local_req->result = default_result;
					cout << "using default result" << endl;
				}
				local_req->cond.notify_one();
			}
		}
	}
}
