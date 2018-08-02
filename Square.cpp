#include "Square.h"

Square::Square(float pX, float pY, float pWidth, float pHeight, bool pVisible = true)
{
	mWidth = pWidth;
	mHeight = pHeight;
	mVertices = {
		pX,pY,
		pX,pY + pHeight,
		pX + pWidth, pY + pHeight,
		pX + pWidth,0
	};	

	mTexcoords = {
		0,1,2,
		2,3,0
	};


	mVisible = pVisible;

}


std::vector<GLfloat> Square::vertices()
{
	return mVertices;
}

std::vector<GLfloat> Square::texcoords()
{
	return mTexcoords;
}

std::vector<GLuint> Square::indices(GLuint pOffset = 0)
{
	return { pOffset + 0, pOffset + 1, pOffset + 2,
		pOffset + 2, pOffset + 3, pOffset + 0 };
}

GLuint Square::texture()
{
	return mTexture;
}

bool Square::visible()
{
	return mVisible;
}

float Square::height()
{
	return mHeight;
}

float Square::width()
{
	return mWidth;
}

Square::~Square()
{
}
