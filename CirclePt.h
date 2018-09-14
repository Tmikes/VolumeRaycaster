#pragma once

#include <vector>

class CirclePt
{
	std::vector<float> mVertices;
	std::vector<float> mTexcoords;
	std::vector<unsigned short> mIndices;
	std::vector<float> mCenter;
public:
	std::vector<float> vertices();
	std::vector<float> texcoords();
	std::vector<float> center();
	std::vector<unsigned short> indices(unsigned int pOffset = 0);
	CirclePt(float pX, float pY, float pR);
	~CirclePt();
};

