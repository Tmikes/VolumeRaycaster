#include <gl/glew.h>
#include "GLMainView.h"


GLdouble AspectRatio;
QVector3D eye(0.5, 0.5, -2), center(0.5, 0.5, 0.5), up(0, 1, 0);

void GLMainView::genFlag(Square pSquare, QVector3D pCol1, QVector3D pCol2, QVector3D pCol3) {
	std::vector<float> colors;
	float r, g, b;
	for (size_t y = 0; y < mHeight; y++)
	{
		for (size_t x = 0; x < mWidth; x++)
		{
			if (x == 0)
			{
				r = pCol1.x; g = pCol1.y; b = pCol1.z;
			}
			else if (x == mWidth / 3)
			{
				r = pCol2.x; g = pCol2.y; b = pCol2.z;
			}
			else if (x == 2 * mWidth / 3) {
				r = pCol3.x; g = pCol3.y; b = pCol3.z;
			}
				colors.push_back(r);
				colors.push_back(g);
				colors.push_back(b);
		}
	}
	int width = (int)(mWidth * pSquare.width());
	int height = (int)(mHeight * pSquare.height());
	glBindTexture(GL_TEXTURE_2D, pSquare.texture());
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
		GL_RGBA, GL_FLOAT, colors.data());
	glBindTexture(GL_TEXTURE_2D, 0);
}

void GLMainView::updateViews()
{
	for (std::vector<Square>::iterator v = mViews.begin(); v != mViews.end(); v++)
	{
		genFlag(*v, QVector3D(0, 0, 1), QVector3D(1, 1, 1), QVector3D(1,0,0));
	}
}



void GLMainView::resetViewer() {
	mTwoViews = false;
	glGenBuffers(1, &mIndexBuf);


}



GLMainView::GLMainView(QWidget *pParent) : QOpenGLWidget(pParent)
{
	// creat the views heeere !!
}

GLMainView::~GLMainView()
{
}

void GLMainView::setData(std::vector<unsigned char> pData)
{
	mData = pData;
}

void GLMainView::setDimensions(int pDimx, int pDimy, int pDimz)
{
	mDimx = pDimx;
	mDimy = pDimy;
	mDimz = pDimz;
}

std::vector<unsigned char> GLMainView::data()
{
	return mData;
}

void GLMainView::initializeGL()
{
	glewInit();
	mProgram = std::make_unique<QOpenGLShaderProgram>(new QOpenGLShaderProgram(this));
	mProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "vertex.glsl");
	mProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "pixel.glsl");
	mPosAttr = mProgram->uniformLocation("vertexPosition_modelspace");
	mTexAttr = mProgram->uniformLocation("vertexUV");
	mMvpUniform = mProgram->uniformLocation("MVP");
	mTexUniform = mProgram->uniformLocation("myTextureSampler");
	mProgram->link();

}



void GLMainView::paintGL()
{
	updateViews();
	// Assemble vertex and indice data for all volumes
	mVertsData.clear();
	mTexData.clear();
	mIndicesData.clear();
	int indicesCount = 0;
	for (std::vector<Square>::iterator view = mViews.begin(); view != mViews.end(); ++view)
	{
		mVertsData.insert(mVertsData.end(), view->vertices().begin(), view->vertices().end());
		mTexData.insert(mTexData.end(), view->texcoords().begin(), view->texcoords().end());
		std::vector<GLuint> currentIndices = view->indices(indicesCount);
		mIndicesData.insert(mIndicesData.end(), currentIndices.begin(), currentIndices.end());
		indicesCount += currentIndices.size();
	}


	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mProgram->bind();
	QMatrix4x4 projectionMatrix;
	projectionMatrix.ortho(0, 1, 0, 1.0 / AspectRatio, 0, 1);
	QMatrix4x4 viewMatrix;
	viewMatrix.lookAt(eye, center, up);
	mProgram->setUniformValue(mMvpUniform, viewMatrix*projectionMatrix);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.05f);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBindBuffer(GL_ARRAY_BUFFER, mPosAttr);
	glBufferData(GL_ARRAY_BUFFER, mVertsData.size() * sizeof(GLfloat), mVertsData.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, mTexAttr);
	glBufferData(GL_ARRAY_BUFFER, mTexData.size() * sizeof(GLfloat), mTexData.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuf);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndicesData.size() * sizeof(GLuint), mIndicesData.data(), GL_STATIC_DRAW);

	mProgram->enableAttributeArray(mPosAttr);
	mProgram->enableAttributeArray(mTexAttr);
	int indiceAt = 0;
	for (std::vector<Square>::iterator view = mViews.begin(); view != mViews.end(); ++view)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, view->texture());
		glUniform1i(mTexUniform, 0);

		glDrawElements(GL_TRIANGLES, view->indices().size(), GL_UNSIGNED_INT, mIndicesData.data() + indiceAt);
		indiceAt += view->indices().size();
	}

	mProgram->disableAttributeArray(mPosAttr);
	mProgram->disableAttributeArray(mTexAttr);
	glFlush();
	mProgram->release();
}

void GLMainView::resizeGL(int pW, int pH) {
	AspectRatio = (GLdouble)(pW) / (GLdouble)(pH);
	glViewport(0, 0, pW, pH);
}

void GLMainView::mouseMoveEvent(QMouseEvent * pEvent)
{
}

void GLMainView::mousePressEvent(QMouseEvent * pEvent)
{
}
