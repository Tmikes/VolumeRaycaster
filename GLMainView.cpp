#include <gl/glew.h>
#include "GLMainView.h"


GLdouble AspectRatio;
QVector3D eye(0.5,0.5,-2), center(0.5, 0.5, 0.5), up(0,1,0);
GLuint indicesBuf;

void GLMainView::resetViewer() {
	mTwoViews = false;
	glGenBuffers(1, &indicesBuf);
	

}



GLMainView::GLMainView(QWidget *pParent) : QOpenGLWidget(pParent)
{
	
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
	mProgram = std::make_unique<QOpenGLShaderProgram>( new QOpenGLShaderProgram(this) );
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
	// Assemble vertex and indice data for all volumes
	mVertsData.clear();
	mTexData.clear();
	mIndicesData.clear();
	int vertcount = 0;
	for (std::vector<Square>::iterator view = mViews.begin(); view != mViews.end(); ++view)
	{
		mVertsData.insert(mVertsData.end(), view->vertices().begin(), view->vertices().end());
		mTexData.insert(mTexData.end(), view->texcoords().begin(), view->texcoords().end());
		mIndicesData.insert(mIndicesData.end(), view->texcoords().begin(), view->texcoords().end());
		vertcount += view->vertices().size();
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mPosAttr);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mVertsData.size() *sizeof(GLfloat) , mVertsData.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mTexAttr);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mTexData.size() * sizeof(GLfloat), mTexData.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);


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

	for (std::vector<Square>::iterator view = mViews.begin() ; view != mViews.end(); ++view)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, view->texture() );
		glUniform1i(mTexUniform, 0);

		glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, view->vertices().data() );
		glVertexAttribPointer(mTexAttr, 2, GL_FLOAT, GL_FALSE, 0, 0;

		glEnableVertexAttribArray(mPosAttr);
		glEnableVertexAttribArray(mTexAttr);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesBuf);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, view->indices().size() * sizeof(unsigned short), view->indices().data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  );
		glDrawElements(GL_TRIANGLES,  view->indices().size(), GL_UNSIGNED_SHORT, 0);
	}
}

void GLMainView::resizeGL(int pW, int pH){
	AspectRatio = (GLdouble)(pW) / (GLdouble)(pH);
	glViewport(0, 0, pW, pH);
}

void GLMainView::mouseMoveEvent(QMouseEvent * pEvent)
{
}

void GLMainView::mousePressEvent(QMouseEvent * pEvent)
{
}
