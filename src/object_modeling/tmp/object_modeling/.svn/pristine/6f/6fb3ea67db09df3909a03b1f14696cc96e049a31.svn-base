#pragma once

class Semaphore
{
private:
    boost::mutex mutex_;
    boost::condition_variable condition_;
    unsigned long count_;

public:
    Semaphore(unsigned long initial_count);
    void notify();
    void wait();
	unsigned long count(); // debug only?
};