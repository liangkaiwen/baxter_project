#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp> // for random
#include <boost/foreach.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // for isnan for readability


#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> // for SURF
#include <opencv2/core/eigen.hpp> // needs to come after eigen

struct compare_greater
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

struct compare_less
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a < b; }
};

using std::cout;
using std::endl;
