#include "VolumeRaycaster.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	VolumeRaycaster w;
	w.show();
	return a.exec();
}
