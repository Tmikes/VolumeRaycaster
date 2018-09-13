#pragma once
#include <gl/glew.h>
#include <QtWidgets/QMainWindow>
#include "ui_VolumeRaycaster.h"

class VolumeRaycaster : public QMainWindow
{
	Q_OBJECT

public:
	VolumeRaycaster(QWidget *parent = Q_NULLPTR);

private:
	Ui::VolumeRaycasterClass ui;

private slots:
	void on_actionOpen_triggered();
	void resizeEvent(QResizeEvent * event);
	void on_tfSlider_valueChanged(int pValue);
	void on_actionDensityMinus_triggered();
	void on_actionDensityAdd_triggered();
	void on_actionLogScale_triggered();
	void on_spinBox_valueChanged(int pValue);
};

void read_binary_data(QByteArray &data, int dimx, int dimy, int dimz, int bytes, int &count, int maxItem,
	std::vector<float> &voxels, double &minV, double &maxV, QString type);
void read_ascii_data(QStringList &in, int dimx, int dimy, int dimz, int bytes, std::vector<float> &voxels, double &minV, double &maxV);
void read_header(QStringList &in , int &dimx, int &dimy, int &dimz, bool &isAsciiData, int &bytes, float &spacingX,
	float &spacingY, float &spacingZ, QString &type);