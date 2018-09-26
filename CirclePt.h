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
	void setVertices(std::vector<float>& const pSource, int pOffset);
	void setTexcoords(std::vector<float>& const pSource, int pOffset);
	void setCenter(std::vector<float>& const pSource, int pOffset);
	std::vector<float> texcoords();
	std::vector<float> center();
	static const int nbrTriangles = 100;
	std::vector<unsigned short> indices(unsigned int pOffset = 0);
	CirclePt(float pX, float pY, float pA, float pB);
	void updateRadius(float pA, float pB);
	~CirclePt();
};

