#pragma once
#include "Square.h"
#include <qopenglwidget.h>
#include <qopenglshaderprogram.h>
#include <QMouseEvent>
#include <memory>
#include <vector>
#include <algorithm>  


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
	float mMinV;
	float mMaxV;
	float mTransferOffset = 0;
	float mDensity = 0;
	std::vector<Square> mViews;

	GLuint mTexView1;
	GLuint mTexView2;
	GLuint mTexOverview;
	GLuint mTexGobal;

	GLuint mPosAttr;
	GLuint mTexAttr;
	GLuint mIndexBuf;
	
	GLuint mMvpUniform;
	GLuint mTexUniform;

	float mTF_offset;
	const float mDensityDelta = 0.01f;
	std::vector<GLfloat> mVertsData;
	std::vector<GLfloat> mTexData;
	std::vector<GLushort> mIndicesData;
	std::vector<float> mData_raw;
	void genFlag(Square& pTexture, QVector3D pCol1, QVector3D pCol2, QVector3D pCol3);
	void raycast(Square& pView);
	bool mTwoViews;
	bool mLoaded;
	//void registerPixelBuffers();
	
protected:
	void mouseMoveEvent(QMouseEvent *pEvent) override;
	void mousePressEvent(QMouseEvent *pEvent) override;
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int pW, int pH) override;

public:
	void setTF_offset(float pOffset);
	void increaseDensity();
	void decreaseDensity();
	void updateViews();
	void resetViewer();
	void setLogScale(bool logScale);
	explicit GLMainView(QWidget *pParent = 0);
	~GLMainView();	
	
	

	//getters
	std::vector<unsigned char> data();
	
	//setters
	void setData(std::vector<float> pDataraw, int pDimx, int pDimy, int pDimz, float pRatiox, float pRatioy, float pRatioz, float minV, float maxV);

};

std::vector<float> scaleVolume(int dimx, int dimy, int dimz, std::vector<float> voxels, float minV, float maxV);
double GenericScaleDouble(double input, double i1, double o1, double i2, double o2);
int myDivUp(int a, int b);