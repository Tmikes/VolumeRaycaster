#pragma once

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
	void on_tfSlider_valueChanged(int pValue);

};
