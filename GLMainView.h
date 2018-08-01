#pragma once
#include <qopenglwidget.h>
#include <qopenglshaderprogram.h>
#include <memory>
#include <vector>
#include "Square.h"

class GLMainView : public QOpenGLWidget
{
	Q_OBJECT
private:
	std::vector<unsigned char> mData;
	std::unique_ptr<QOpenGLShaderProgram>  mProgram;
	int mFrame;
	int mDimx;
	int mDimy;
	int mDimz;
	
	std::vector<Square> mViews;

	GLuint mTexView1;
	GLuint mTexView2;
	GLuint mTexOverview;
	GLuint mTexGobal;

	GLuint mPosAttr;
	GLuint mTexAttr;
	GLuint mPbo;
	GLuint mMvpUniform;
	GLuint mTexUniform;

	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mTexData;
	std::vector<unsigned int> mIndicesData;
	
	bool mTwoViews;
public:
	void resetViewer();
	explicit GLMainView(QWidget *pParent = 0);
	~GLMainView();	
	void initializeGL();
	void paintGL();
	void resizeGL(int pW, int pH);
	void mouseMoveEvent(QMouseEvent *pEvent);
	void mousePressEvent(QMouseEvent *pEvent);

	//getters
	std::vector<unsigned char> data();
	
	//setters
	void setData(std::vector<unsigned char> pData);
	void GLMainView::setDimensions(int pDimx, int pDimy, int pDimz);

};

