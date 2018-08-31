#include "Square.h"
#include <qmatrix.h>

QMatrix4x4 rotations;
Square::Square(float pX, float pY, float pWidth, float pHeight, float pTexWidth, float pTexHeight, QVector3D pTranslation , bool pVisible )
{
	mWidth = pWidth;
	mHeight = pHeight;
	mVertices = {
		pX,pY,
		pX,pY + pHeight,
		pX + pWidth, pY + pHeight,
		pX + pWidth,pY
	};	

	mTexcoords = {
		0,0,
		0,1,
		1,1,
		1,0
	};

	mVisible = pVisible;
	// create pixel buffer object for display
	glGenBuffers(1, &mPbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mPbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pTexWidth*pTexHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glGenTextures(1, &mTexture);
	glBindTexture(GL_TEXTURE_2D, mTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pTexWidth, pTexHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL /* empty data*/);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	

	cudaError t = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, mPbo, cudaGraphicsMapFlagsWriteDiscard);
	checkCudaErrors(t);
	mTexWidth = pTexWidth;
	mTexHeight = pTexHeight;
	mCenter = pTranslation;
	mTranslation = QVector3D(0,0,3);
	mViewMatrix.translate(mTranslation);
}


std::vector<GLfloat> Square::vertices()
{
	return mVertices;
}

std::vector<GLfloat> Square::texcoords()
{
	return mTexcoords;
}

std::vector<GLushort> Square::indices(GLuint pOffset )
{
	return { (GLushort)(pOffset + 0), (GLushort)(pOffset + 1), (GLushort)(pOffset + 2),
		(GLushort)(pOffset + 2), (GLushort)(pOffset + 3),(GLushort)(pOffset + 0 )};
}

void Square::updateViewMatrix(QVector2D pAngles)
{
	
	QMatrix4x4 rotX;// rotXa fromAxisAndAngle(rotations.toRotationMatrix() * QVector3D(1, 0, 0), pAngles.x());
	QMatrix4x4 rotY;// = QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0), pAngles.y());

	//rotations = Q * rotations;
	//rotations.normalize();
	//rotations = rotations * QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0), pAngles.y()).normalized();
	//rotations.normalize();
	mViewMatrix.translate(-mTranslation);
	mViewMatrix.rotate(-pAngles.x(), QVector3D(0, 1, 0));
	mViewMatrix.rotate(-pAngles.y(), QVector3D(1, 0, 0));
	mViewMatrix.translate(mTranslation);

	rotations.rotate(-pAngles.x(), QVector3D(0, 1, 0));
	rotations.rotate(-pAngles.y(), QVector3D(1, 0, 0));
	//mViewMatrix = rotX * mViewMatrix;
	//mViewMatrix = rotY * mViewMatrix;
}

std::vector<float> Square::viewMatrix()
{
	/*QMatrix4x4 tmp;
	tmp.translate(mTranslation);
	QMatrix4x4 rot;
	rot.rotate(rotations);
	mViewMatrix =  rot * tmp;*/
	float* data = mViewMatrix.data();
	return { 
				data[0],data[4],data[8],data[12],
				data[1],data[5],data[9],data[13],
				data[2],data[6],data[10],data[14], 
	};
}

GLuint Square::texture()
{
	return mTexture;
}

GLuint Square::pbo()
{
	return mPbo;
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

float Square::texWidth()
{
	return mTexWidth;
}

float Square::texHeight()
{
	return mTexHeight;
}

void Square::zoom(float pDelta)
{
	mTranslation += QVector3D(0,0, pDelta);
	//mViewMatrix.setToIdentity();
	mViewMatrix = rotations;
	mViewMatrix.translate(mTranslation);
}



Square::~Square()
{
	//checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
}
