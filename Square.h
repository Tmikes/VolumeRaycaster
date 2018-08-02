#pragma once
#include <gl/glew.h>
#include <vector>
class Square
{
private :
	GLuint mTexture;
	std::vector<GLfloat> mVertices;
	std::vector<GLfloat> mTexcoords;
	float mWidth;
	float mHeight;
	bool mVisible;
public:

	Square(float pX, float pY, float pWidth, float pHeight, bool pVisible = true);
	std::vector<GLfloat> vertices();
	std::vector<GLfloat> texcoords();
	std::vector<GLuint> indices(GLuint pOffset = 0);
	GLuint texture();
	bool visible();
	float height();
	float width();
	~Square();
};

