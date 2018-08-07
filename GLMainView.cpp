#include <gl/glew.h>
#include "volumehelper.h"
#include "GLMainView.h"
#include <fstream>


GLdouble AspectRatio;
QVector3D eye(0.5, 0.5, -2), target(0.5, 0.5, 0.5), up(0, 1, 0);
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
void GLMainView::genFlag(Square& pSquare, QVector3D pCol1, QVector3D pCol2, QVector3D pCol3) {
	std::vector<GLubyte> colors;
	float r, g, b;
	int width = (int)(mWidth * pSquare.width());
	int height = (int)(mHeight * pSquare.height());
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			if (x <  width / 3)
			{
				r = pCol1.x(); g = pCol1.y(); b = pCol1.z();
			}
			else if (x < 2* width / 3)
			{
				r = pCol2.x(); g = pCol2.y(); b = pCol2.z();
			}
			else {
				r = pCol3.x(); g = pCol3.y(); b = pCol3.z();
			}
				colors.push_back((GLubyte)r);
				colors.push_back((GLubyte)g);
				colors.push_back((GLubyte)b);
				colors.push_back((GLubyte)255);
		}
		std::vector<int>  a = { 1,2,3,4,5,6,7,8,9,10 },
			b = { 1,1,1,1,1,1,1,1,1,1 }, c(10);
		addWithCuda(c, a, b, 10);
		int d = c[0];
	}
	
	glBindTexture(GL_TEXTURE_2D, pSquare.texture());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, colors.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

//void GLMainView::registerPixelBuffers()
//{
//	checkCudaErrors(cudaGraphicsUnregisterResource(mCuda_pbo_resource));
//	for (std::vector<Square>::iterator view  = mViews.begin();  view != mViews.end(); view++)
//	{
//		// register this buffer object with CUDA
//		//struct cudaGraphicsResource *tmpRessource = cuda_pbo_resource.get();
//		
//	}
//	
//
//}



void GLMainView::updateViews()
{
	for (std::vector<Square>::iterator v = mViews.begin(); v != mViews.end(); v++)
	{
		genFlag((*v), QVector3D(0, 0, 255), QVector3D(255, 255, 255), QVector3D(255,0,0));
	}

	


}



void GLMainView::resetViewer() {
	mTwoViews = false;
	mViews.push_back(Square(0, 0, 0.5, 0.5, mWidth, mHeight));
	mLoaded = false;
	//mViews.push_back(Square(0.5, 0.5, 0.5, 0.5, mWidth, mHeight));


}


void GLMainView::raycast(Square& pView)
{
	copyInvViewMatrix(pView.viewMatrix());
	dim3 blockSize(8,8,1);
	dim3 gridSize(iDivUp(pView.texWidth(), blockSize.x), iDivUp(pView.texHeight(), blockSize.y));
	// map PBO to get CUDA device pointer
	uint *d_output;
	
	// map PBO to get CUDA device pointer
	cudaError err =  cudaGraphicsMapResources(1, &(pView.cuda_pbo_resource), 0);
	checkCudaErrors(err);
//	checkCudaErrors(cudaGraphicsMapResources(1, &(pView.cuda_pbo_resource), 0));


	int a = 1 + 5;
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, pView.cuda_pbo_resource);
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	cudaMemset(d_output, 0, pView.texWidth()*pView.texHeight() * sizeof(uint));

	

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, pView.texWidth(), pView.texHeight(), mDensity, mTransferOffset);

//	getLastCudaError("kernel failed");

	cudaGraphicsUnmapResources(1, &pView.cuda_pbo_resource, 0);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// copy from pbo to texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pView.pbo());
	glBindTexture(GL_TEXTURE_2D, pView.texture());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pView.texWidth(), pView.texHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}


GLMainView::GLMainView(QWidget *pParent) : QOpenGLWidget(pParent)
{
	// creat the views heeere !!
	
	mWidth = mHeight = 1024;
}

GLMainView::~GLMainView()
{
	freeCudaBuffers();

}

void GLMainView::setData(std::vector<unsigned char> pData, int pDimx, int pDimy, int pDimz)
{
	mData = pData;
	mDimx = pDimx;
	mDimy = pDimy;
	mDimz = pDimz;
	initCuda(mData, mDimx, mDimy, mDimz);
	mLoaded = true;
}



std::vector<unsigned char> GLMainView::data()
{
	return mData;
}

void GLMainView::initializeGL()
{
	glewInit();
	mProgram = std::make_unique<QOpenGLShaderProgram>(new QOpenGLShaderProgram(this));
	mProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "view.v.glsl");
	mProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "view.f.glsl");
	mProgram->link();
	mPosAttr = mProgram->attributeLocation("vertexPosition_modelspace");
	mTexAttr = mProgram->attributeLocation("vertexUV");
	mMvpUniform = mProgram->uniformLocation("MVP");
	mTexUniform = mProgram->uniformLocation("myTextureSampler");
	//cuda
	int devID = gpuGetMaxGflopsDeviceId();
	cudaGLSetGLDevice(devID);

	glGenBuffers(1, &mIndexBuf);
	resetViewer();
	updateViews();


}



void GLMainView::paintGL()
{
	
	// Assemble vertex and indice data for all volumes
	mVertsData.clear();
	mTexData.clear();
	mIndicesData.clear();
	int vertscount = 0;
	for (std::vector<Square>::iterator view = mViews.begin(); view != mViews.end(); ++view)
	{
		if (mLoaded)
		{
			raycast(*view);
		}
		
		std::vector<GLfloat> currentVerts = view->vertices(), currentTexcoords = view->texcoords();
		mVertsData.insert(mVertsData.end(), currentVerts.begin(), currentVerts.end());
		mTexData.insert(mTexData.end(), currentTexcoords.begin(), currentTexcoords.end());
		std::vector<GLushort> currentIndices = view->indices(vertscount);
		mIndicesData.insert(mIndicesData.end(), currentIndices.begin(), currentIndices.end());
		vertscount += currentVerts.size()/2;
	}


	glClearColor(1.0, 1.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mProgram->bind();
	QMatrix4x4 projectionMatrix;
	projectionMatrix.ortho(0, 1, 0, 1.0 , -2, 2);
	QMatrix4x4 viewMatrix;
	viewMatrix.translate(0, 0, -1);
	mProgram->setUniformValue(mMvpUniform, projectionMatrix*viewMatrix);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.05f);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	
	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, mVertsData.data() );

	glVertexAttribPointer(mTexAttr, 2, GL_FLOAT, GL_FALSE, 0, mTexData.data());

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuf);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndicesData.size() * sizeof(GLushort), mIndicesData.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(mPosAttr);
	glEnableVertexAttribArray(mTexAttr);
	int indiceAt = 0;
	for (std::vector<Square>::iterator view = mViews.begin(); view != mViews.end(); ++view)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, view->texture());
		glUniform1i(mTexUniform, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuf);
		glDrawElements(GL_TRIANGLES, view->indices().size(), GL_UNSIGNED_SHORT, (void *) (indiceAt * sizeof(GLushort)));
		indiceAt += view->indices().size();
	}
	glDisableVertexAttribArray(mTexAttr);
	glDisableVertexAttribArray(mPosAttr);
	//glFlush();
	mProgram->release();
}

void GLMainView::resizeGL(int pW, int pH) {
	AspectRatio = (GLdouble)(pW) / (GLdouble)(pH);
	glViewport(0, 0, pW, pH);
}

int oldX, oldY;
void GLMainView::mouseMoveEvent(QMouseEvent * pEvent)
{
	QVector2D rotationAngles(pEvent->x() - oldX, pEvent->y() - oldY);
	mViews[0].updateViewMatrix(rotationAngles);
	oldX = pEvent->x();
	oldY = pEvent->y();
}

void GLMainView::mousePressEvent(QMouseEvent * pEvent)
{
	oldX = pEvent->x();
	oldY = pEvent->y();
}
