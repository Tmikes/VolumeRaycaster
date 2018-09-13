#pragma once
#include <vector>

class CirclePt
{
	std::vector<float> mVertices;
	std::vector<float> mTexcoords;
	std::vector<float> mCenter;
public:
	CirclePt();
	~CirclePt();
};

