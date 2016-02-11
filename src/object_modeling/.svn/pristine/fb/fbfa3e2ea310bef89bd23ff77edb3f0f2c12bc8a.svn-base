#pragma once

#include "cll.h"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <vector>
#include <map>

class OpenCLKernelsBuilder
{
public:
	OpenCLKernelsBuilder(CL& cl, fs::path const& source_path, std::vector<std::string> const& kernel_names, bool debug, bool fast_math);

	const std::map<std::string, cl::Kernel> & getKernelMap() const;

protected:

	// shouldn't need this anymore:
	cl::Kernel getKernel(std::string const& name);

	std::map<std::string, cl::Kernel> kernel_name_map;
};

