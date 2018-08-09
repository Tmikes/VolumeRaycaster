#include "VolumeRaycaster.h"
#include <qmessagebox.h>
#include <qfiledialog.h>

VolumeRaycaster::VolumeRaycaster(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(ui.actionOpen->triggered),
		this, SLOT(on_actionOpen_triggered));
	connect(ui.tfSlider, SIGNAL(ui.tfSlider->valueChanged),
		this, SLOT(on_tfSlider_valueChanged));
}

void VolumeRaycaster::on_actionOpen_triggered() {
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open a volume"), "",
		tr("All Files (*);;VTK file (*.vtk)"));
	if (fileName.isEmpty())
		return;
	else {

		QFile file(fileName);

		if (!file.open(QIODevice::ReadOnly)) {
			QMessageBox::information(this, tr("Unable to open file"),
				file.errorString());
			return;
		}
		QDataStream in(&file);
		int dimx = 32, dimy = 32, dimz = 32;
		std::vector<unsigned char> data(dimx*dimy*dimz);
		in.setVersion(QDataStream::Qt_4_5);
		int i = file.size();
		in.readRawData((char*)data.data(), i);
		ui.openGLWidget->setData(data, dimx, dimy, dimz);

	}

}

void VolumeRaycaster::on_tfSlider_valueChanged(int pValue)
{
	float min = ui.tfSlider->minimum(), max = ui.tfSlider->maximum();
	ui.openGLWidget->setTF_offset((pValue - min) / (max - min));
}
