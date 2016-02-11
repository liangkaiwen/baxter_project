#pragma once

#include <string>

class ROSTimestamp {
public:
	// 0,0
	ROSTimestamp();

	// expects ("   1234567890.123456  ", 3)
	ROSTimestamp(std::string const& s);

	std::string str() const;

protected:
	friend std::ostream & operator<<(std::ostream &os, const ROSTimestamp& ros_timestamp);

	// members
	int sec;
	int nsec;
};

std::ostream & operator<<(std::ostream &os, const ROSTimestamp& ros_timestamp);
