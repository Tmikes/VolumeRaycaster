#include "GLTransferFunction.h"


float wRatio;
void GLTransferFunction::mouseMoveEvent(QMouseEvent * pEvent)
{
}

void GLTransferFunction::mousePressEvent(QMouseEvent * pEvent)
{
}

void GLTransferFunction::wheelEvent(QWheelEvent * event)
{
}

void GLTransferFunction::resetTF() {
	mColors = {
		//----------
		{
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 1.0f, 0.84f, 0.45f, 0.5f, },
			{ 1.0f, 0.84f, 0.45f, 0.5f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 1.0f, }
		},
		//----------		
		{
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 1.0f, 0.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 1.0f, }
		},
		//----------		
		{
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 1.0f, 0.84f, 0.45f, 0.0f, },
			{ 0.0f, 1.0f, 0.0f, 0.0f, },
			{ 0.0f, 1.0f, 0.0f, 0.0f, },
			{ 0.0f, 1.0f, 0.0f, 0.0f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 0.5f, },
			{ 0.0f, 0.0f, 1.0f, 1.0f, }
		}
	};


}

void GLTransferFunction::initializeGL()
{
	mIndex = 0;
	mProgram = std::make_unique<QOpenGLShaderProgram>(new QOpenGLShaderProgram(this));
	mProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "circle.v.glsl");
	mProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "circle.f.glsl");
	mProgram->link();
	mPosAttr = mProgram->attributeLocation("vertexPosition_modelspace");
	mTexAttr = mProgram->attributeLocation("vertexUV");
	mCenterAttr = mProgram->attributeLocation("vertexCenter");
	mMvpUniform = mProgram->uniformLocation("MVP");
	mTexUniform = mProgram->uniformLocation("myTextureSampler");
	resetTF();
	glGenTextures(1, &mTexture);
	glBindTexture(GL_TEXTURE_1D, mTexture);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, mColors[mIndex].size(), 0, GL_RGBA, GL_FLOAT, mColors[mIndex].data());
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_1D, 0);
	for (int i = 0; i < mColors[mIndex].size(); i++)
	{
		float x = i / (float)(mColors[mIndex].size() - 1);
		float y = mColors[mIndex][i].w;
		mCircles.push_back(CirclePt(x,y,0.05f,0.05f));
	}
	//CirclePt c0(0.5,0.5,0.3);
	//mCircles.push_back(c0);
}

void GLTransferFunction::paintGL()
{
	// Assemble vertex and indice data for all volumes
	mVertsData.clear();
	mTexData.clear();
	mIndicesData.clear();
	int vertscount = 0;
	for (std::vector<CirclePt>::iterator circle = mCircles.begin(); circle != mCircles.end(); ++circle)
	{
		std::vector<GLfloat> currentVerts = circle->vertices(), currentTexcoords = circle->texcoords(),
			currentCenter = circle->center();
		mVertsData.insert(mVertsData.end(), currentVerts.begin(), currentVerts.end());
		mTexData.insert(mTexData.end(), currentTexcoords.begin(), currentTexcoords.end());
		mCenterData.insert(mCenterData.end(), currentCenter.begin(), currentCenter.end());
		std::vector<GLushort> currentIndices = circle->indices(vertscount);
		mIndicesData.insert(mIndicesData.end(), currentIndices.begin(), currentIndices.end());
		vertscount += currentVerts.size() / 2;
	}

	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mProgram->bind();
	QMatrix4x4 projectionMatrix;
	projectionMatrix.ortho(0, 1, 0, 1, -2, 2);
	QMatrix4x4 viewMatrix;
	viewMatrix.translate(0, 0, -1);
	mProgram->setUniformValue(mMvpUniform, projectionMatrix*viewMatrix);
	/*glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.05f);*/

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	glVertexAttribPointer(mPosAttr, 2, GL_FLOAT, GL_FALSE, 0, mVertsData.data());
	glVertexAttribPointer(mCenterAttr, 2, GL_FLOAT, GL_FALSE, 0, mCenterData.data());
	glVertexAttribPointer(mTexAttr, 2, GL_FLOAT, GL_FALSE, 0, mTexData.data());

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuf);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndicesData.size() * sizeof(GLushort), mIndicesData.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(mPosAttr);
	glEnableVertexAttribArray(mTexAttr);
	glEnableVertexAttribArray(mCenterAttr);
	int indiceAt = 0;
	for (std::vector<CirclePt>::iterator circle = mCircles.begin(); circle != mCircles.end(); ++circle)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, mTexture);
		glUniform1i(mTexUniform, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuf);
		glDrawElements(GL_TRIANGLES, circle->indices().size(), GL_UNSIGNED_SHORT, (void *)(indiceAt * sizeof(GLushort)));
		indiceAt += circle->indices().size();
	}
	glDisableVertexAttribArray(mTexAttr);
	glDisableVertexAttribArray(mPosAttr);
	glDisableVertexAttribArray(mCenterAttr);

	glDisable(GL_BLEND);
	glFlush();
	mProgram->release();
}

void GLTransferFunction::resizeGL(int pW, int pH)
{

	wRatio = (GLdouble)(pW) / (GLdouble)(pH);

	glViewport(0, 0, pW, pH);
}

GLTransferFunction::GLTransferFunction(QWidget * pParent) : QOpenGLWidget(pParent)
{
}

GLTransferFunction::~GLTransferFunction()
{
}
