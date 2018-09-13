#pragma once
#include <qopenglwidget.h>
#include <qopenglshaderprogram.h>
#include "volumehelper.h"
#include <QMouseEvent>
#include <memory>
#include <vector>
#include <algorithm>  


class GLTransferFunction : public QOpenGLWidget
{
	Q_OBJECT
private:
	std::vector<std::vector<float4>> mColors;

	std::unique_ptr<QOpenGLShaderProgram>  mProgram;

	GLuint mPosAttr;
	GLuint mTexAttr;
	GLuint mCenterAttr;
	GLuint mIndexBuf;

	GLuint mMvpUniform;
	GLuint mTexUniform;

	
	const float mDensityDelta = 0.01f;
	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mTexData;
	std::vector<GLushort> mIndicesData;

protected:
	void mouseMoveEvent(QMouseEvent *pEvent) override;
	void mousePressEvent(QMouseEvent *pEvent) override;
	void wheelEvent(QWheelEvent *event) override;
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int pW, int pH) override;

public:
	explicit GLTransferFunction(QWidget *pParent = 0);
	~GLTransferFunction();
};
