#include "GLTransferFunction.h"

void GLTransferFunction::mouseMoveEvent(QMouseEvent * pEvent)
{
}

void GLTransferFunction::mousePressEvent(QMouseEvent * pEvent)
{
}

void GLTransferFunction::wheelEvent(QWheelEvent * event)
{
}

void GLTransferFunction::initializeGL()
{
	mProgram = std::make_unique<QOpenGLShaderProgram>(new QOpenGLShaderProgram(this));
	mProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "view.v.glsl");
	mProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "view.f.glsl");
	mProgram->link();
	mPosAttr = mProgram->attributeLocation("vertexPosition_modelspace");
	mTexAttr = mProgram->attributeLocation("vertexUV");
	mMvpUniform = mProgram->uniformLocation("MVP");
	mTexUniform = mProgram->uniformLocation("myTextureSampler");
}

void GLTransferFunction::paintGL()
{
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void GLTransferFunction::resizeGL(int pW, int pH)
{
}

GLTransferFunction::GLTransferFunction(QWidget * pParent) : QOpenGLWidget(pParent)
{
}

GLTransferFunction::~GLTransferFunction()
{
}
