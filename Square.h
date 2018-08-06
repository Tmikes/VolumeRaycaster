#pragma once
#include <gl/glew.h>
#include <vector>
#include <qvector2d.h>
#include <qvector3d.h>
#include <qmatrix4x4.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

//// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

class Square
{
private :
	GLuint mTexture;
	std::vector<GLfloat> mVertices;
	std::vector<GLfloat> mTexcoords;
	float mWidth;
	float mHeight;
	bool mVisible;
	QMatrix4x4 mViewMatrix;
public:

	Square(float pX, float pY, float pWidth, float pHeight, float pTexWidth, float pTexHeight, bool pVisible = true);
	std::vector<GLfloat> vertices();
	std::vector<GLfloat> texcoords();
	std::vector<GLushort> indices(GLuint pOffset = 0);
	void updateViewMatrix(QVector2D pAngles);
	std::vector<float> viewMatrix();
	QVector3D mTranslation;
	GLuint texture();
	bool visible();
	float height();
	float width();
	~Square();
};

