#pragma once
#include <gl/glew.h>
#include <qopenglwidget.h>
#include <qopenglshaderprogram.h>
#include "volumehelper.h"
#include <QMouseEvent>
#include <memory>
#include <vector>
#include <algorithm>  
#include "CirclePt.h"


class GLTransferFunction : public QOpenGLWidget
{
	Q_OBJECT
private:
	std::vector<std::vector<float4>> mColors;
	std::vector<float> mCenters;
	std::unique_ptr<QOpenGLShaderProgram>  mProgram;
	float mIndex;
	GLuint mPosAttr;
	GLuint mTexAttr;
	GLuint mCenterAttr;
	GLuint mIndexBuf;
	GLuint mTexture;
	GLuint mMvpUniform;
	GLuint mTexUniform;
	
	
	const float mDensityDelta = 0.01f;
	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mCenterData;
	std::vector<GLfloat> mTexData;
	std::vector<GLushort> mIndicesData;
	std::vector<CirclePt> mCircles;
	
protected:
	void mouseMoveEvent(QMouseEvent *pEvent) override;
	void mousePressEvent(QMouseEvent *pEvent) override;
	void wheelEvent(QWheelEvent *event) override;
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int pW, int pH) override;

public:
	void resetTF();
	explicit GLTransferFunction(QWidget *pParent = 0);
	~GLTransferFunction();
};
