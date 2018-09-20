#define _USE_MATH_DEFINES
#include <cmath>
#include "CirclePt.h"
#include <algorithm>
#include <execution>
#include <cctype>


std::vector<float> CirclePt::vertices()
{
	return mVertices;
}

std::vector<float> CirclePt::texcoords()
{
	return mTexcoords;
}

std::vector<float> CirclePt::center()
{
	return mCenter;
}

std::vector<unsigned short> CirclePt::indices(unsigned int pOffset)
{
	std::vector<unsigned short> tmp(mIndices);
	std::for_each(std::execution::par_unseq, tmp.begin(), tmp.end(), [pOffset](unsigned short& item)
	{
		item += pOffset;
	});
	return tmp;
}

CirclePt::CirclePt(float pX, float pY, float pA, float pB)
{
	int segments = 100; 
	for (int n = 0; n < segments; n++) {
		float const t1 = 2 * M_PI * (float)n / (float)segments;
		float const t2 = 2 * M_PI * (float)(n+1) / (float)segments;
		
		// vertices
		mVertices.push_back(pX);
		mVertices.push_back(pY);
		mVertices.push_back(pX + sin(t1) * pA);
		mVertices.push_back(pY + cos(t1) * pB);
		mVertices.push_back(pX + sin(t2) * pA);
		mVertices.push_back(pY + cos(t2) * pB);

		// texcoords
		mTexcoords.push_back(pX);
		mTexcoords.push_back(pY);
		mTexcoords.push_back(pX + sin(t1) * pA);
		mTexcoords.push_back(pY + cos(t1) * pB);
		mTexcoords.push_back(pX + sin(t2) * pA);
		mTexcoords.push_back(pY + cos(t2) * pB);

		// indices
		mIndices.push_back(n * 3 + 0);
		mIndices.push_back(n * 3 + 1);
		mIndices.push_back(n * 3 + 2);


		//center

		mCenter.push_back(pX);
		mCenter.push_back(pY);
		mCenter.push_back(pX);
		mCenter.push_back(pY);
		mCenter.push_back(pX);
		mCenter.push_back(pY);
	}
}

void CirclePt::updateRadius(float pA, float pB)
{

}


CirclePt::~CirclePt()
{
}
