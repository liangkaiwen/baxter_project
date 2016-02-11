#include "StdAfx.h"
#include "Semaphore.h"

Semaphore::Semaphore(unsigned long initial_count)
	: count_(initial_count)
{}

void Semaphore::notify()
{
	boost::mutex::scoped_lock lock(mutex_);
	++count_;
	condition_.notify_one();
}

void Semaphore::wait()
{
	boost::mutex::scoped_lock lock(mutex_);
	while(!count_)
		condition_.wait(lock);
	--count_;
}

unsigned long Semaphore::count()
{
	boost::mutex::scoped_lock lock(mutex_);
	return count_;
}