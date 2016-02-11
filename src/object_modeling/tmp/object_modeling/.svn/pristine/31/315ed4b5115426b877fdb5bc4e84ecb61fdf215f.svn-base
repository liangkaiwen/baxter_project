#pragma once

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <vector>
#include <map>

#include "cll.h"
#include "cl_util.h"

class OpenCLAllKernels
{
public:
	OpenCLAllKernels(CL& cl, fs::path const& source_path, bool debug = false, bool fast_math = false);

	CL& getCL() {return cl_;}

	cl::Kernel getKernel(std::string const& name) const;

protected:
	CL& cl_;
	std::map<std::string, cl::Kernel> kernel_name_map_;

};

