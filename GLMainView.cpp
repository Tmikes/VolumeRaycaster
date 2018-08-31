#include <gl/glew.h>
#include "volumehelper.h"
#include "GLMainView.h"
#include <fstream>
#include <algorithm>
#include <execution>


GLdouble AspectRatio;
QVector3D eye(0.5, 0.5, -2), target(0.5, 0.5, 0.5), up(0, 1, 0);
int myDivUp(int a, int b)
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
	
	/*glBindTexture(GL_TEXTURE_2D, pSquare.texture());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, colors.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);*/

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



void GLMainView::setTF_offset(float pOffset)
{
	mTF_offset = pOffset;
	update();
}

void GLMainView::increaseDensity()
{
	mDensity += mDensityDelta;
	update();
}

void GLMainView::decreaseDensity()
{
	mDensity = std::max(0.0f, mDensity - mDensityDelta);
	update();
}

void GLMainView::updateViews()
{
	for (std::vector<Square>::iterator v = mViews.begin(); v != mViews.end(); v++)
	{
		genFlag((*v), QVector3D(0, 0, 255), QVector3D(255, 255, 255), QVector3D(255,0,0));
	}

	


}



void GLMainView::resetViewer() {
	mTwoViews = false;
	mViews.push_back(Square(0, 0, 1, 1, mWidth, mHeight));
	mLoaded = false;
	//mViews.push_back(Square(0.5, 0.5, 0.5, 0.5, mWidth, mHeight));


}

void GLMainView::setLogScale(bool withLogScale)
{
	std::vector<float> raw =  scaleVolume(mDimx, mDimy, mDimz, mData_raw, mMinV, mMaxV);
	//blurData(mData_raw, mData_raw,mDimx, mDimy, mDimz,3);

	std::vector<byte> tmp(raw.begin(), raw.end());
	mData = tmp;
	logScale(mData, withLogScale);
	update();
}


void GLMainView::raycast(Square& pView)
{
	cudaGraphicsResource *cuda_pbo_resource;
	GLuint pbo = 0;
	//if (pbo)
	//{
	//	// unregister this buffer object from CUDA C
	//	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

	//	// delete old buffer
	//	glDeleteBuffers(1, &pbo);
	//	
	//}

	//
	//glGenBuffers(1, &pbo);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, mWidth*mHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//GLuint tex;
	//// create texture for display
	//glGenTextures(1, &tex);
	//glBindTexture(GL_TEXTURE_2D, tex);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glBindTexture(GL_TEXTURE_2D, 0);


	/*cudaError t = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pView.pbo(), cudaGraphicsMapFlagsWriteDiscard);
	checkCudaErrors(t);
*/
	

	copyInvViewMatrix(pView.viewMatrix());
	dim3 blockSize(8,8,1);
	dim3 gridSize(myDivUp(pView.texWidth(), blockSize.x), myDivUp(pView.texHeight(), blockSize.y));
	// map PBO to get CUDA device pointer
	uint *d_output;
	
	// map PBO to get CUDA device pointer
	cudaError err =  cudaGraphicsMapResources(1, &( pView.cuda_pbo_resource), 0);
	checkCudaErrors(err);
//	checkCudaErrors(cudaGraphicsMapResources(1, &(pView.cuda_pbo_resource), 0));


	int a = 1 + 5;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, pView.cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, pView.texWidth()*pView.texHeight() * 4));

	

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, pView.texWidth(), pView.texHeight(), mDensity, mTF_offset, float3({ (float)mDimx,(float)mDimy, (float)mDimz }));

	getLastCudaError("kernel failed");

	

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pView.cuda_pbo_resource, 0));
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	////GL.Color3(Color.White);
	//glClear(GL_COLOR_BUFFER_BIT);
	

	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// copy from pbo to texture
	
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pView.pbo());
	glBindTexture(GL_TEXTURE_2D, pView.texture());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pView.texWidth(), pView.texHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	std::vector<unsigned char> tmp(pView.texWidth()*pView.texHeight() * 4);
	
	glBindTexture(GL_TEXTURE_2D, pView.texture());
	
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
	glBindTexture(GL_TEXTURE_2D, 0);
	//glBindTexture(GL_TEXTURE_2D, 0);
}


GLMainView::GLMainView(QWidget *pParent) : QOpenGLWidget(pParent)
{
	// creat the views heeere !!
	
	mWidth = mHeight = 1024;
	mDensity = 0.05f;
}

GLMainView::~GLMainView()
{
	freeCudaBuffers();

}

void GLMainView::setData(std::vector<float> pDataraw, int pDimx, int pDimy, int pDimz,
	float pRatiox , float pRatioy, float pRatioz, float minV, float maxV)
{
	
	mDimx = pDimx;
	mDimy = pDimy;
	mDimz = pDimz;
	mMinV = minV;
	mMaxV = maxV;
	mData_raw = pDataraw;
	pDataraw = scaleVolume(mDimx, mDimy, mDimz, pDataraw, minV, maxV);
	blurData(pDataraw, pDataraw,mDimx, mDimy, mDimz,3);

	std::vector<byte> tmp(pDataraw.begin(), pDataraw.end());
	mData = tmp;
	initCuda(mData, mDimx, mDimy, mDimz);
	mLoaded = true;
}


std::vector<float> scaleVolume(int dimx, int dimy, int dimz, std::vector<float> voxels, float minV, float maxV)
{
	std::for_each(std::execution::par_unseq, voxels.begin(), voxels.end(), [minV,maxV](float& item)
	{
		//do stuff with item
		double scaleV = GenericScaleDouble(item, minV, 0, maxV, 255);
		if (scaleV < 0)
		{
			scaleV = 0;
		}
		item = round(scaleV);
	});
	return voxels;
}

double GenericScaleDouble(double input, double i1, double o1, double i2, double o2)
{
	if (i2 == i1) return (o1 + o2) / 2.0; //Arbitrary choice, but wrong!!!

	return (input - i1) * (o2 - o1) / (i2 - i1) + o1;
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


	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mProgram->bind();
	QMatrix4x4 projectionMatrix;
	projectionMatrix.ortho(0, 1, 0, 1.0 , -2, 2);
	QMatrix4x4 viewMatrix;
	viewMatrix.translate(0, 0, -1);
	mProgram->setUniformValue(mMvpUniform, projectionMatrix*viewMatrix);
	/*glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.05f);*/

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

	glDisable(GL_BLEND);
	glFlush();
	mProgram->release();
}

void GLMainView::resizeGL(int pW, int pH) {
	AspectRatio = (GLdouble)(pW) / (GLdouble)(pH);
	glViewport(0, 0, pW, pH);
}

int oldX, oldY;
void GLMainView::mouseMoveEvent(QMouseEvent * pEvent)
{
	if(pEvent->buttons() & Qt::LeftButton) {
		QVector2D rotationAngles(pEvent->x() - oldX, pEvent->y() - oldY);
		mViews[0].updateViewMatrix(rotationAngles);
		oldX = pEvent->x();
		oldY = pEvent->y();
		this->update();
	}

	
}

void GLMainView::mousePressEvent(QMouseEvent * pEvent)
{
	oldX = pEvent->x();
	oldY = pEvent->y();
}
