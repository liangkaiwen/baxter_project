// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "targetver.h"

// automatic:
#include <stdio.h>
//#include <tchar.h>

// stl
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <list>
#include <limits>
#include <string>
#include <exception>
#include <stdexcept>
#include <set>
#include <map>

// boost
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/thread.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> // for SURF








