#pragma once
#include <gl/glew.h>
#include <vector>
#include <qvector2d.h>
#include <qvector3d.h>
#include <qmatrix4x4.h>
#include <memory>

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
	float mTexWidth;
	float mTexHeight;
	GLuint mPbo;
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
	GLuint pbo();
	cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
	bool visible();
	float height();
	float width();
	float texWidth();
	float texHeight();
	~Square();
};

