#pragma once

#include <cstddef>
#include <vector>

class DisjointSet
{
public:
	DisjointSet(size_t elements);
	~DisjointSet(void);

	int find(int element);
	int size(int element);
	// returns the new parent component of both
	int connect(int element_1, int element_2);

protected:
	std::vector<int> v;
};

