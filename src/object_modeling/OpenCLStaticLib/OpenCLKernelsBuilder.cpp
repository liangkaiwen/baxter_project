#include "stdafx.h"
#include "OpenCLKernelsBuilder.h"

#include "cl_util.h"

using std::cout;
using std::endl;

OpenCLKernelsBuilder::OpenCLKernelsBuilder(CL& cl, fs::path const& source_path, std::vector<std::string> const& kernel_names, bool debug, bool fast_math)
{
	try {
		std::ifstream sourceFile(source_path.string().c_str());
		if (!sourceFile) {
			std::string error_string = "Bad source file: " + source_path.string();
			cout << error_string << endl;
			throw std::runtime_error (error_string.c_str());
		}
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		// pass sourceFilenameFullPath to enable debugging
		// this is actually a silly way to do it...since you ahve to print the source path out here if you want to see it
		cout << "source_path: " << source_path << endl;
		fs::path sourceFilenameFullPath;
		if (debug) {
			sourceFilenameFullPath = fs::system_complete(source_path);
		}
		std::string additional_args;
		if (fast_math) {
			additional_args += " -cl-fast-relaxed-math";
		}
		int which_program = cl.loadProgram(sourceCode, sourceFilenameFullPath.string(), additional_args);
		if (which_program >= 0) {
			for (std::vector<std::string>::const_iterator iter = kernel_names.begin(); iter != kernel_names.end(); ++iter) {
				kernel_name_map[*iter] = cl::Kernel(cl.programs[which_program], iter->c_str());
			}
		}
		else {
			throw std::runtime_error("which_program < 0");
		}
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw std::runtime_error("failed to load source code somehow");
	}
}

cl::Kernel OpenCLKernelsBuilder::getKernel(std::string const& name)
{
	return kernel_name_map[name];
}

const std::map<std::string, cl::Kernel> & OpenCLKernelsBuilder::getKernelMap() const
{
	return kernel_name_map;
}
