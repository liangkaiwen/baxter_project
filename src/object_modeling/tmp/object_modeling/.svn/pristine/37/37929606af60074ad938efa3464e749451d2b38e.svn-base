#include "disjoint_set.h"


DisjointSet::DisjointSet(size_t elements)
{
	v.assign(elements, -1);
}


DisjointSet::~DisjointSet(void)
{
}


int DisjointSet::find(int element)
{
	int result = element;
	if (v[element] >= 0) {
		result = find(v[element]);
		v[element] = result;
	}
	return result;
}

int DisjointSet::size(int element)
{
	int top = find(element);
	return -v[top];
}

int DisjointSet::connect(int element_1, int element_2)
{
	int top_1 = find(element_1);
	int top_2 = find(element_2);
	if (top_1 != top_2) {
		int size_1 = size(top_1);
		int size_2 = size(top_2);
		if (size_1 > size_2) {
			v[top_2] = top_1;
			v[top_1] = -(size_1 + size_2);
			return top_1;
		}
		else {
			v[top_1] = top_2;
			v[top_2] = -(size_1 + size_2);
			return top_2;
		}
	}
	else {
		return top_1; // both are the same
	}
}