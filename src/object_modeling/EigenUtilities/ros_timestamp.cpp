#include "ros_timestamp.h"

#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

ROSTimestamp::ROSTimestamp()
	: sec(0),
	nsec(0)
{}

ROSTimestamp::ROSTimestamp(std::string const& s)
	: sec(0),
	nsec(0)
{
	std::vector<std::string> tokens;
    std::string trimmed = boost::trim_copy(s);
    boost::split(tokens, trimmed, boost::is_any_of("."));
	if (tokens.size() == 2) {
		sec = atoi(tokens[0].c_str());
		nsec = atoi(tokens[1].c_str());
		int nsec_factor = 9 - tokens[1].length();
		for (int i = 0; i < nsec_factor; ++i) nsec *= 10;
	}
	else {
		throw std::runtime_error("bad ROSTimestamp string");
	}
}

std::string ROSTimestamp::str() const
{
	return (boost::format("%010d.%09d") % sec % nsec).str();
}

std::ostream & operator<<(std::ostream &os, const ROSTimestamp& ros_timestamp)
{
	return os << ros_timestamp.str();
}
