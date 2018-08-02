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
	int mWidth;
	int mHeight;
	
	std::vector<Square> mViews;

	GLuint mTexView1;
	GLuint mTexView2;
	GLuint mTexOverview;
	GLuint mTexGobal;

	GLuint mPosAttr;
	GLuint mTexAttr;
	GLuint mIndexBuf;
	GLuint mPbo;
	GLuint mMvpUniform;
	GLuint mTexUniform;

	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mTexData;
	std::vector<unsigned int> mIndicesData;
	void genFlag(Square pTexture, QVector3D pCol1, QVector3D pCol2, QVector3D pCol3);
	bool mTwoViews;
public:
	void updateViews();

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

