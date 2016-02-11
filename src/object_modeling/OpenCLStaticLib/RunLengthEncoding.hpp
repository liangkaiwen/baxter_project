#pragma once

#include <vector>

template <typename T>
void runLengthEncode(std::vector<T> const& v, std::vector<std::pair<size_t, T> > & result)
{
	result.clear();
	if (v.empty()) return;
	std::pair<size_t, T> current(1, v.front());
	for (size_t i = 1; i < v.size(); ++i) {
		T const& t = v[i];
		if (t == current.second) {
			++current.first;
		}
		else {
			result.push_back(current);
			current = std::make_pair(1, t);
		}
	}
	result.push_back(current);
}

template <typename T>
void runLengthDecode(std::vector<std::pair<size_t, T> > const& v_pairs, std::vector<T> & result)
{
	result.clear();
	for (size_t i = 0; i < v_pairs.size(); ++i) {
		std::pair<size_t, T> const& p = v_pairs[i];
		for (size_t c = 0; c < p.first; ++c) {
			result.push_back(p.second);
		}
	}
}

