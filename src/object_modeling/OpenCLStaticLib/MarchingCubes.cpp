// File Name: CIsoSurface.cpp
// Last Modified: 5/8/2000
// Author: Raghavendra Chandrashekara (based on source code provided
// by Paul Bourke and Cory Gene Bloyd)
// Email: rc99@doc.ic.ac.uk, rchandrashekara@hotmail.com
//
// Heavily modified by Peter

#include "stdafx.h"
#include <math.h>
#include "MarchingCubes.h"

#include <iostream>
using std::cout;
using std::endl;

template <class T> const unsigned int MarchingCubes<T>::m_edgeTable[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
template <class T> const unsigned int MarchingCubes<T>::m_triTable[256][16] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};
#pragma GCC diagnostic pop


template <class T> MarchingCubes<T>::MarchingCubes(const MarchingCubesParams & params)
	: params_(params)
{
}


#if 0
template <class T> void MarchingCubes<T>::generateSurface(const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField, const Eigen::Array3i & num_cells, const Eigen::Array3f & cell_sizes, Mesh & result_mesh, std::vector<bool> & result_valid)
{
    // not needed once we have correct blah blah
    unsigned int nPointsInXDirection = num_cells[0];
    unsigned int nPointsInSlice = nPointsInXDirection*num_cells[1];

    // do need this (was member)
    IDToVertexPlusT id_to_vertex_plus;

    // use result_mesh.triangles for triangles always?
    result_mesh.vertices.clear();
    result_mesh.triangles.clear();

	// Generate isosurface.
    for (unsigned int z = 0; z < num_cells[2] - 1; z++)
        for (unsigned int y = 0; y < num_cells[1] - 1; y++)
            for (unsigned int x = 0; x < num_cells[0] - 1; x++) {
				// Calculate table lookup index from those
				// vertices which are below the isolevel.
				unsigned int tableIndex = 0;
                if (ptScalarField[z*nPointsInSlice + y*nPointsInXDirection + x] < 0)
					tableIndex |= 1;
                if (ptScalarField[z*nPointsInSlice + (y+1)*nPointsInXDirection + x] < 0)
					tableIndex |= 2;
                if (ptScalarField[z*nPointsInSlice + (y+1)*nPointsInXDirection + (x+1)] < 0)
					tableIndex |= 4;
                if (ptScalarField[z*nPointsInSlice + y*nPointsInXDirection + (x+1)] < 0)
					tableIndex |= 8;
                if (ptScalarField[(z+1)*nPointsInSlice + y*nPointsInXDirection + x] < 0)
					tableIndex |= 16;
                if (ptScalarField[(z+1)*nPointsInSlice + (y+1)*nPointsInXDirection + x] < 0)
					tableIndex |= 32;
                if (ptScalarField[(z+1)*nPointsInSlice + (y+1)*nPointsInXDirection + (x+1)] < 0)
					tableIndex |= 64;
                if (ptScalarField[(z+1)*nPointsInSlice + y*nPointsInXDirection + (x+1)] < 0)
					tableIndex |= 128;

				// Now create a triangulation of the isosurface in this
				if (m_edgeTable[tableIndex] != 0) {
					if (m_edgeTable[tableIndex] & 8) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 3);
                        unsigned int id = getEdgeID(num_cells, x, y, z, 3);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
					}
					if (m_edgeTable[tableIndex] & 1) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 0);
                        unsigned int id = getEdgeID(num_cells, x, y, z, 0);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
					}
					if (m_edgeTable[tableIndex] & 256) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 8);
                        unsigned int id = getEdgeID(num_cells, x, y, z, 8);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
					}
					
                    if (x == num_cells[0] - 2) {
						if (m_edgeTable[tableIndex] & 4) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 2);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 2);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
						if (m_edgeTable[tableIndex] & 2048) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 11);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 11);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
					}
                    if (y == num_cells[1] - 2) {
						if (m_edgeTable[tableIndex] & 2) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 1);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 1);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
						if (m_edgeTable[tableIndex] & 512) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 9);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 9);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
					}
                    if (z == num_cells[2] - 2) {
						if (m_edgeTable[tableIndex] & 16) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 4);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 4);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
						if (m_edgeTable[tableIndex] & 128) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 7);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 7);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
					}
                    if ((x==num_cells[0] - 2) && (y==num_cells[1] - 2))
						if (m_edgeTable[tableIndex] & 1024) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 10);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 10);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
                    if ((x==num_cells[0] - 2) && (z==num_cells[2] - 2))
						if (m_edgeTable[tableIndex] & 64) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 6);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 6);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
                    if ((y==num_cells[1] - 2) && (z==num_cells[2] - 2))
						if (m_edgeTable[tableIndex] & 32) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_sizes, x, y, z, 5);
                            unsigned int id = getEdgeID(num_cells, x, y, z, 5);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
						}
					
					for (unsigned int i = 0; m_triTable[tableIndex][i] != -1; i += 3) {
                        Triangle triangle;
						unsigned int pointID0, pointID1, pointID2;
                        pointID0 = getEdgeID(num_cells, x, y, z, m_triTable[tableIndex][i]);
                        pointID1 = getEdgeID(num_cells, x, y, z, m_triTable[tableIndex][i+1]);
                        pointID2 = getEdgeID(num_cells, x, y, z, m_triTable[tableIndex][i+2]);
                        triangle[0] = pointID0;
                        triangle[1] = pointID1;
                        triangle[2] = pointID2;
                        result_mesh.triangles.push_back(triangle);
					}
				}
			}
	
    renameVerticesAndTriangles(id_to_vertex_plus, result_mesh, result_valid);
    calculateNormals(result_mesh);
}
#endif
// rather, in terms of pieces:
template <class T> void MarchingCubes<T>::generateSurface(const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField, const Eigen::Array3i & num_cells, const Eigen::Array3f & cell_sizes, Mesh & result_mesh, std::vector<bool> & result_valid)
{
    IDToVertexPlusT id_to_vertex_plus;
	IDTriangleVector id_triangle_vector;
	IDToDebugT id_debug;

	// don't really need to do this here anymore
    result_mesh.vertices.clear();
    result_mesh.triangles.clear();

    generateSurfaceForBlock(ptScalarField, ptWeightField, ptColorField, true, true, true, num_cells, Eigen::Array3i(0,0,0), num_cells, cell_sizes, id_to_vertex_plus, id_triangle_vector, id_debug);

    finalizeMesh(id_to_vertex_plus, id_triangle_vector, id_debug, result_mesh, result_valid);
}


// the "core" functionality
// will add to current_triangle_list and current_vertex_map
template <class T> void MarchingCubes<T>::generateSurfaceForBlock(
        const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField,
		bool last_x, bool last_y, bool last_z,
        const Eigen::Array3i & global_num_cells, const Eigen::Array3i & cell_offset, const Eigen::Array3i & num_cells, const Eigen::Array3f & cell_sizes,
        IDToVertexPlusT & id_to_vertex_plus, IDTriangleVector & id_triangle_list, IDToDebugT & id_debug
		)
{
    // not needed once we have correct blah blah
    unsigned int nPointsInXDirection = num_cells[0];
    unsigned int nPointsInSlice = nPointsInXDirection*num_cells[1];

    // Generate isosurface.
    for (unsigned int z = 0; z < num_cells[2] - 1; z++)
        for (unsigned int y = 0; y < num_cells[1] - 1; y++)
            for (unsigned int x = 0; x < num_cells[0] - 1; x++) {
                // Calculate table lookup index from those
                // vertices which are below the isolevel.
                unsigned int tableIndex = 0;
                if (ptScalarField[z*nPointsInSlice + y*nPointsInXDirection + x] < 0)
                    tableIndex |= 1;
                if (ptScalarField[z*nPointsInSlice + (y+1)*nPointsInXDirection + x] < 0)
                    tableIndex |= 2;
                if (ptScalarField[z*nPointsInSlice + (y+1)*nPointsInXDirection + (x+1)] < 0)
                    tableIndex |= 4;
                if (ptScalarField[z*nPointsInSlice + y*nPointsInXDirection + (x+1)] < 0)
                    tableIndex |= 8;
                if (ptScalarField[(z+1)*nPointsInSlice + y*nPointsInXDirection + x] < 0)
                    tableIndex |= 16;
                if (ptScalarField[(z+1)*nPointsInSlice + (y+1)*nPointsInXDirection + x] < 0)
                    tableIndex |= 32;
                if (ptScalarField[(z+1)*nPointsInSlice + (y+1)*nPointsInXDirection + (x+1)] < 0)
                    tableIndex |= 64;
                if (ptScalarField[(z+1)*nPointsInSlice + y*nPointsInXDirection + (x+1)] < 0)
                    tableIndex |= 128;

                // Now create a triangulation of the isosurface in this
                if (m_edgeTable[tableIndex] != 0) {
                    if (m_edgeTable[tableIndex] & 8) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 3);
                        IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 3);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                    }
                    if (m_edgeTable[tableIndex] & 1) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 0);
                        IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 0);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                    }
                    if (m_edgeTable[tableIndex] & 256) {
                        VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 8);
                        IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 8);
                        id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                    }

                    if (last_x && x == num_cells[0] - 2) {
                        if (m_edgeTable[tableIndex] & 4) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 2);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 2);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                        if (m_edgeTable[tableIndex] & 2048) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 11);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 11);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                    }
                    if (last_y && y == num_cells[1] - 2) {
                        if (m_edgeTable[tableIndex] & 2) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 1);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 1);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                        if (m_edgeTable[tableIndex] & 512) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 9);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 9);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                    }
                    if (last_z && z == num_cells[2] - 2) {
                        if (m_edgeTable[tableIndex] & 16) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 4);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 4);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                        if (m_edgeTable[tableIndex] & 128) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 7);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 7);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                    }
                    if (last_x && last_y && (x==num_cells[0] - 2) && (y==num_cells[1] - 2))
                        if (m_edgeTable[tableIndex] & 1024) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 10);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 10);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                    if (last_x && last_z && (x==num_cells[0] - 2) && (z==num_cells[2] - 2))
                        if (m_edgeTable[tableIndex] & 64) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 6);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 6);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }
                    if (last_y && last_z && (y==num_cells[1] - 2) && (z==num_cells[2] - 2))
                        if (m_edgeTable[tableIndex] & 32) {
                            VertexPlus vp = calculateIntersection(ptScalarField, ptWeightField, ptColorField, num_cells, cell_offset, cell_sizes, x, y, z, 5);
                            IDT id = getEdgeID(global_num_cells, cell_offset, x, y, z, 5);
                            id_to_vertex_plus.insert(typename IDToVertexPlusT::value_type(id, vp));
                        }

                    for (unsigned int i = 0; m_triTable[tableIndex][i] != -1; i += 3) {
                        //Triangle triangle;
						IDTriangle triangle;
                        IDT pointID0 = getEdgeID(global_num_cells, cell_offset, x, y, z, m_triTable[tableIndex][i]);
                        IDT pointID1 = getEdgeID(global_num_cells, cell_offset, x, y, z, m_triTable[tableIndex][i+1]);
                        IDT pointID2 = getEdgeID(global_num_cells, cell_offset, x, y, z, m_triTable[tableIndex][i+2]);
                        triangle[0] = pointID0;
                        triangle[1] = pointID1;
                        triangle[2] = pointID2;
                        id_triangle_list.push_back(triangle);

                        // stupid debug (always put vertices that are EVER missing into the id_debug structure)
                        for (int i = 0; i < 3; ++i) {
                            if (id_to_vertex_plus.find(triangle[i]) == id_to_vertex_plus.end()) {
                                //cout << "(maybe ok) missing vertex " << triangle[i] << " for " << Eigen::Array3i(x,y,z).transpose() << " with last_xyz: " << last_x << last_y << last_z << " and cell_offset: " << cell_offset.transpose() << endl;
                                DebugT debug;
                                debug.xyz = Eigen::Array3i(x,y,z);
                                debug.cell_offset = cell_offset;
                                debug.last_bool = Eigen::Array3i(last_x, last_y, last_z);
                                id_debug[triangle[i]] = debug;
                            }
                        }
                    }
                }
            }
}

template <class T> void MarchingCubes<T>::finalizeMesh(IDToVertexPlusT & id_to_vertex_plus, IDTriangleVector const& id_triangle_vector, IDToDebugT & id_debug, Mesh & result_mesh, std::vector<bool> & result_valid)
{
	renameVerticesAndTriangles(id_to_vertex_plus, id_triangle_vector, id_debug, result_mesh, result_valid);
    calculateNormals(result_mesh);
}

template <class T> typename MarchingCubes<T>::IDT MarchingCubes<T>::getEdgeID(const Eigen::Array3i & global_num_cells, const Eigen::Array3i & cell_offset, unsigned int nX, unsigned int nY, unsigned int nZ, unsigned int nEdgeNo)
{
	switch (nEdgeNo) {
	case 0:
        return getVertexID(global_num_cells, cell_offset, nX, nY, nZ) + 1;
	case 1:
        return getVertexID(global_num_cells, cell_offset, nX, nY + 1, nZ);
	case 2:
        return getVertexID(global_num_cells, cell_offset, nX + 1, nY, nZ) + 1;
	case 3:
        return getVertexID(global_num_cells, cell_offset, nX, nY, nZ);
	case 4:
        return getVertexID(global_num_cells, cell_offset, nX, nY, nZ + 1) + 1;
	case 5:
        return getVertexID(global_num_cells, cell_offset, nX, nY + 1, nZ + 1);
	case 6:
        return getVertexID(global_num_cells, cell_offset, nX + 1, nY, nZ + 1) + 1;
	case 7:
        return getVertexID(global_num_cells, cell_offset, nX, nY, nZ + 1);
	case 8:
        return getVertexID(global_num_cells, cell_offset, nX, nY, nZ) + 2;
	case 9:
        return getVertexID(global_num_cells, cell_offset, nX, nY + 1, nZ) + 2;
	case 10:
        return getVertexID(global_num_cells, cell_offset, nX + 1, nY + 1, nZ) + 2;
	case 11:
        return getVertexID(global_num_cells, cell_offset, nX + 1, nY, nZ) + 2;
	default:
		// Invalid edge no.
		return -1;
	}
}

template <class T> typename MarchingCubes<T>::IDT MarchingCubes<T>::getVertexID(const Eigen::Array3i & global_num_cells, const Eigen::Array3i & cell_offset, unsigned int nX, unsigned int nY, unsigned int nZ)
{
    //return 3*((IDT)nZ*((IDT)num_cells[1] + 1)*((IDT)num_cells[0] + 1) + (IDT)nY*((IDT)num_cells[0] + 1) + (IDT)nX);
	Eigen::Array3i this_cell_offset = cell_offset + Eigen::Array3i(nX, nY, nZ);
    return 3*((IDT)this_cell_offset[2]*((IDT)global_num_cells[1])*((IDT)global_num_cells[0]) + (IDT)this_cell_offset[1]*((IDT)global_num_cells[0]) + (IDT)this_cell_offset[0]);
}

template <class T> typename MarchingCubes<T>::VertexPlus MarchingCubes<T>::calculateIntersection(
	const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField,
	const Eigen::Array3i & num_cells, const Eigen::Array3i & cell_offset, const Eigen::Array3f & cell_sizes,
    unsigned int nX, unsigned int nY, unsigned int nZ, unsigned int nEdgeNo)
{
	unsigned int v1x = nX, v1y = nY, v1z = nZ;
	unsigned int v2x = nX, v2y = nY, v2z = nZ;
	
	switch (nEdgeNo)
	{
	case 0:
		v2y += 1;
		break;
	case 1:
		v1y += 1;
		v2x += 1;
		v2y += 1;
		break;
	case 2:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		break;
	case 3:
		v1x += 1;
		break;
	case 4:
		v1z += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 5:
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 6:
		v1x += 1;
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2z += 1;
		break;
	case 7:
		v1x += 1;
		v1z += 1;
		v2z += 1;
		break;
	case 8:
		v2z += 1;
		break;
	case 9:
		v1y += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 10:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 11:
		v1x += 1;
		v2x += 1;
		v2z += 1;
		break;
	}

    Eigen::Vector4f p1, p2;
    p1.head<3>() = (Eigen::Array3f(v1x, v1y, v1z) * cell_sizes).matrix();
    p1[3] = 1;
    p2.head<3>() = (Eigen::Array3f(v2x, v2y, v2z) * cell_sizes).matrix();
    p2[3] = 1;
	
    // this will change...
    unsigned int nPointsInXDirection = num_cells[0];
    unsigned int nPointsInSlice = nPointsInXDirection*num_cells[1];
    T val1 = ptScalarField[v1z*nPointsInSlice + v1y*nPointsInXDirection + v1x];
    T val2 = ptScalarField[v2z*nPointsInSlice + v2y*nPointsInXDirection + v2x];

    // fill out result_vertex
    VertexPlus result;
    result.v.p = interpolate(p1, p2, val1, val2);
	// only on non-null color, do it?
	// the idea being we could easily go back and do without color
	if (ptColorField) {
#if 0
        // this works, but I think I can get map to work
		unsigned int index_1 = 4 * (v1z*nPointsInSlice + v1y*nPointsInXDirection + v1x);
		Eigen::Array4ub c1(ptColorField[index_1], ptColorField[index_1+1], ptColorField[index_1+2], ptColorField[index_1+3]);
		unsigned int index_2 = 4 * (v2z*nPointsInSlice + v2y*nPointsInXDirection + v2x);
		Eigen::Array4ub c2(ptColorField[index_2], ptColorField[index_2+1], ptColorField[index_2+2], ptColorField[index_2+3]);
#endif

#if 0
        // this works:
        Eigen::Array4ub c1 = Eigen::Map<const Eigen::Array<unsigned char, 4, 1> >(&ptColorField[ 4 * (v1z*nPointsInSlice + v1y*nPointsInXDirection + v1x)]);
        Eigen::Array4ub c2 = Eigen::Map<const Eigen::Array<unsigned char, 4, 1> >(&ptColorField[ 4 * (v2z*nPointsInSlice + v2y*nPointsInXDirection + v2x)]);
#endif

        // this works!!:
        Eigen::Array4ub c1 = Eigen::Map<const Eigen::Array4ub >(&ptColorField[ 4 * (v1z*nPointsInSlice + v1y*nPointsInXDirection + v1x)]);
        Eigen::Array4ub c2 = Eigen::Map<const Eigen::Array4ub >(&ptColorField[ 4 * (v2z*nPointsInSlice + v2y*nPointsInXDirection + v2x)]);

		result.v.c = interpolateColor(p1, p2, result.v.p, c1, c2);
	}
	// awkwardly, only now apply offset to v.p
    Eigen::Vector4f & result_v_p = result.v.p;
    result_v_p.head<3>() += (cell_offset.cast<float>() * cell_sizes).matrix();


    float weight1 = ptWeightField[v1z*nPointsInSlice + v1y*nPointsInXDirection + v1x];
    float weight2 = ptWeightField[v2z*nPointsInSlice + v2y*nPointsInXDirection + v2x];
    static const float epsilon = 1e-6; // zero should probably be fine too
    result.valid = (weight1 > epsilon && weight2 > epsilon);

	// max valid value
    //result.valid = result.valid && (fabs(val1) < max_value_value && fabs(val2) < max_value_value);
	// or too far apart?
	result.valid = result.valid && (params_.max_value <= 0 || fabs(val1 - val2) <= params_.max_value);

	return result;
}

template <class T> Eigen::Vector4f MarchingCubes<T>::interpolate(Eigen::Vector4f const& p1, Eigen::Vector4f const& p2, T tVal1, T tVal2)
{
    Eigen::Vector4f interpolation;
	float mu;

    //mu = float((m_tIsoLevel - tVal1))/(tVal2 - tVal1);
    mu = float((- tVal1))/(tVal2 - tVal1);
    interpolation = p1 + mu * (p2 - p1);

	return interpolation;
}

template <class T> Eigen::Array4ub MarchingCubes<T>::interpolateColor(Eigen::Vector4f const& p1, Eigen::Vector4f const& p2, Eigen::Vector4f const& p_interpolated, Eigen::Array4ub const& c1, Eigen::Array4ub const& c2)
{
	// could do this smarter
	float alpha = (p_interpolated - p1).norm() / (p2 - p1).norm();

	Eigen::Array4f result_float = (1 - alpha) * c1.cast<float>() + alpha * c2.cast<float>();

	// from interpolateColorForMesh (sort of...)
	// should do this color float blending stuff just once...
	const static Eigen::Array4f add_round(0.5,0.5,0.5,0.5);
	const static Eigen::Array4ub min_array(0,0,0,0);
	const static Eigen::Array4ub max_array(255,255,255,255);

	Eigen::Array4f result_to_cast = result_float + add_round;
	Eigen::Array4ub result = result_to_cast.cast<unsigned char>();
#undef max
#undef min
	result = result.max(min_array);
	result = result.min(max_array);

	return result;
}


template <class T> void MarchingCubes<T>::renameVerticesAndTriangles(IDToVertexPlusT & id_to_vertex_plus, const IDTriangleVector & id_triangle_vector, const IDToDebugT & id_debug, Mesh & result_mesh, std::vector<bool> & result_valid)
{
	unsigned int next_small_id = 0;
    typename IDToVertexPlusT::iterator map_iter = id_to_vertex_plus.begin();

    result_mesh.vertices.clear();
	result_mesh.triangles.clear();
    result_valid.clear();

    // Rename vertices
    // also fill out actual result while we are here
	// todo: size once?  or at least reserve?
    while (map_iter != id_to_vertex_plus.end()) {
        map_iter->second.new_id = next_small_id;
		result_mesh.vertices.push_back(map_iter->second.v);
        result_valid.push_back(map_iter->second.valid);
        next_small_id++;
        map_iter++;
	}

	// Now rename triangles.
    //TriangleVector::iterator triangle_iter = input_and_result_mesh.triangles.begin();
	IDTriangleVector::const_iterator triangle_iter = id_triangle_vector.begin();
    while (triangle_iter != id_triangle_vector.end()) {
		Triangle output_triangle;
		for (unsigned int i = 0; i < 3; i++) {
			// stupid debug
			{
				IDT vertex = (*triangle_iter)[i];
				if (id_to_vertex_plus.find(vertex) == id_to_vertex_plus.end()) {
                    cout << "BAD missing vertex: " << vertex << endl;
					DebugT debug = id_debug.find(vertex)->second;
					cout << "debug xyz: " << debug.xyz.transpose() << " cell_offset: " << debug.cell_offset.transpose() << " last: " << debug.last_bool.transpose() << endl;
				}
			}

            unsigned int newID = id_to_vertex_plus[(*triangle_iter)[i]].new_id;
            output_triangle[i] = newID;
		}
		result_mesh.triangles.push_back(output_triangle);
        triangle_iter++;
	}

#if 0
	// stupid debug
	cout << "stupid debug remove" << endl;
	for (int i = 0; i < result_mesh.triangles.size(); ++i) {
		if ((result_mesh.triangles[i].array() < 0).any()) {
			cout << "Negative: " << result_mesh.triangles[i].transpose() << endl;
		}
		if ((result_mesh.triangles[i].array() > next_small_id-1).any()) {
			cout << "bigger than " << (next_small_id-1) << " : " << result_mesh.triangles[i].transpose() << endl;
		}
    }
#endif
}

template <class T> void MarchingCubes<T>::calculateNormals(Mesh & input_and_result_mesh)
{

    /*
	// Set all normals to 0.
	for (unsigned int i = 0; i < m_nNormals; i++) {
		m_pvec3dNormals[i][0] = 0;
		m_pvec3dNormals[i][1] = 0;
		m_pvec3dNormals[i][2] = 0;
	}

	// Calculate normals.
	for (unsigned int i = 0; i < m_nTriangles; i++) {
		VECTOR3D vec1, vec2, normal;
		unsigned int id0, id1, id2;
		id0 = m_piTriangleIndices[i*3];
		id1 = m_piTriangleIndices[i*3+1];
		id2 = m_piTriangleIndices[i*3+2];
		vec1[0] = m_ppt3dVertices[id1][0] - m_ppt3dVertices[id0][0];
		vec1[1] = m_ppt3dVertices[id1][1] - m_ppt3dVertices[id0][1];
		vec1[2] = m_ppt3dVertices[id1][2] - m_ppt3dVertices[id0][2];
		vec2[0] = m_ppt3dVertices[id2][0] - m_ppt3dVertices[id0][0];
		vec2[1] = m_ppt3dVertices[id2][1] - m_ppt3dVertices[id0][1];
		vec2[2] = m_ppt3dVertices[id2][2] - m_ppt3dVertices[id0][2];
		normal[0] = vec1[2]*vec2[1] - vec1[1]*vec2[2];
		normal[1] = vec1[0]*vec2[2] - vec1[2]*vec2[0];
		normal[2] = vec1[1]*vec2[0] - vec1[0]*vec2[1];
		m_pvec3dNormals[id0][0] += normal[0];
		m_pvec3dNormals[id0][1] += normal[1];
		m_pvec3dNormals[id0][2] += normal[2];
		m_pvec3dNormals[id1][0] += normal[0];
		m_pvec3dNormals[id1][1] += normal[1];
		m_pvec3dNormals[id1][2] += normal[2];
		m_pvec3dNormals[id2][0] += normal[0];
		m_pvec3dNormals[id2][1] += normal[1];
		m_pvec3dNormals[id2][2] += normal[2];
	}

	// Normalize normals.
	for (unsigned int i = 0; i < m_nNormals; i++) {
		float length = sqrt(m_pvec3dNormals[i][0]*m_pvec3dNormals[i][0] + m_pvec3dNormals[i][1]*m_pvec3dNormals[i][1] + m_pvec3dNormals[i][2]*m_pvec3dNormals[i][2]);
		m_pvec3dNormals[i][0] /= length;
		m_pvec3dNormals[i][1] /= length;
		m_pvec3dNormals[i][2] /= length;
	}
    */

    for (MeshVertexVector::iterator iter = input_and_result_mesh.vertices.begin() ; iter != input_and_result_mesh.vertices.end(); ++iter) {
        iter->n = Eigen::Vector4f::Zero();
    }

    for (TriangleVector::iterator iter = input_and_result_mesh.triangles.begin() ; iter != input_and_result_mesh.triangles.end(); ++iter) {
        // try to stick to vector4 instead?
        MeshVertex & v0 = input_and_result_mesh.vertices[(*iter)[0]];
        MeshVertex & v1 = input_and_result_mesh.vertices[(*iter)[1]];
        MeshVertex & v2 = input_and_result_mesh.vertices[(*iter)[2]];
        Eigen::Vector3f ray_1 = (v1.p - v0.p).head<3>();
        Eigen::Vector3f ray_2 = (v2.p - v0.p).head<3>();
        // assume clockwise?
        Eigen::Vector3f normal = ray_1.cross(ray_2);
        v0.n.head<3>() += normal;
        v1.n.head<3>() += normal;
        v2.n.head<3>() += normal;
    }

    // normalize
    for (MeshVertexVector::iterator iter = input_and_result_mesh.vertices.begin() ; iter != input_and_result_mesh.vertices.end(); ++iter) {
        iter->n.normalize();
    }
}

/*
template class CIsoSurface<short>;
template class CIsoSurface<unsigned short>;
template class CIsoSurface<float>;
*/

// instantiate?
template class MarchingCubes<float>;
