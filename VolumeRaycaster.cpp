#include "VolumeRaycaster.h"
#include <qmessagebox.h>

VolumeRaycaster::VolumeRaycaster(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(ui.actionOpen->triggered),
		this, SLOT(on_actionOpen_triggered));
}

void VolumeRaycaster::on_actionOpen_triggered() {
	/*QMessageBox msgbox;
	msgbox.setText("sum of numbers are...." );
	msgbox.exec();*/
}