#pragma once
#include "Square.h"
#include <qopenglwidget.h>
#include <qopenglshaderprogram.h>
#include <QMouseEvent>
#include <memory>
#include <vector>


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
	struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
	GLuint mMvpUniform;
	GLuint mTexUniform;

	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mTexData;
	std::vector<GLushort> mIndicesData;
	void genFlag(Square pTexture, QVector3D pCol1, QVector3D pCol2, QVector3D pCol3);
	void raycast(Square pView);
	bool mTwoViews;
	void initPixelBuffer();

protected:
	void mouseMoveEvent(QMouseEvent *pEvent) override;
	void mousePressEvent(QMouseEvent *pEvent) override;
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int pW, int pH) override;

public:
	void updateViews();
	void resetViewer();
	explicit GLMainView(QWidget *pParent = 0);
	~GLMainView();	
	
	

	//getters
	std::vector<unsigned char> data();
	
	//setters
	void setData(std::vector<unsigned char> pData);
	void GLMainView::setDimensions(int pDimx, int pDimy, int pDimz);

};

