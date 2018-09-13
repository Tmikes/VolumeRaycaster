#include "VolumeRaycaster.h"
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <limits.h>
#include <cstring>
#include <algorithm>
#include <iomanip> // setprecision
#include <sstream> // stringstream
#include <QTextCodec>
VolumeRaycaster::VolumeRaycaster(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(ui.actionOpen->triggered),
		this, SLOT(on_actionOpen_triggered));
	connect(ui.actionDensityAdd, SIGNAL(ui.actionDensityAdd->triggered),
		this, SLOT(on_actionDensityAdd_triggered));
	connect(ui.actionDensityMinus, SIGNAL(ui.actionDensityMinus->triggered),
		this, SLOT(on_actionDensityMinus_triggered));
	connect(ui.tfSlider, SIGNAL(ui.tfSlider->valueChanged),
		this, SLOT(on_tfSlider_valueChanged));
	connect(ui.actionLogScale, SIGNAL(ui.actionLogScale->triggered),
		this, SLOT(on_actionLogScale_triggered));
	connect(ui.spinBox, SIGNAL(ui.spinBox->valueChanged),
		this, SLOT(on_spinBox_valueChanged));
	
}

void VolumeRaycaster::resizeEvent(QResizeEvent* event) {
	QMainWindow::resizeEvent(event);
	QSize old = event->oldSize() ; 
	if (old.height() != -1 )
	{
		QSize delta = event->size() - old;
		ui.openGLWidget->resize(ui.openGLWidget->size() + delta);
		ui.tfSlider->resize(ui.tfSlider->width() + delta.width(), ui.tfSlider->height());
		ui.tfSlider->move(QPoint(ui.tfSlider->pos().x(), ui.tfSlider->pos().y() + delta.height()));
		ui.minLabel->move(QPoint(ui.minLabel->pos().x(), ui.minLabel->pos().y() + delta.height()));
		ui.maxLabel->move(QPoint(ui.maxLabel->pos().x(), ui.maxLabel->pos().y() + delta.height()));
		ui.spinBox->move(QPoint(ui.spinBox->pos().x(), ui.spinBox->pos().y() + delta.height()));
	}
	
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

		int dimx = 0, dimy = 0, dimz = 0;

		//in.setVersion(QDataStream::Qt_5_1);
		int file_size = file.size();
		int densityStats[256];
 
		/* ---- Reading the header ---- */
		float spacingX = 1, spacingY = 1, spacingZ = 1;
		int bytes = 1;
		bool isAsciiData = true;
		QString type;
		QByteArray filedata = file.readAll();
		QString dataAsString(filedata);
		//QTextStream reader(&file);
		QStringList lines = dataAsString.split(QRegExp("[\r\n]"), QString::SkipEmptyParts);
		read_header(lines, dimx, dimy, dimz, isAsciiData, bytes, spacingX, spacingY, spacingZ, type);

		int maxItem = dimx * dimy * dimz;
		std::vector<unsigned char> data(maxItem);
		std::vector<float> voxels(maxItem);

		double minV = std::numeric_limits<double>::max();
		double maxV = std::numeric_limits<double>::min();
		//in.readRawData((char*)data.data(), file_size);
		file.seek(file.size() - dimx * dimy * dimz * bytes);
		
		int count = 0;
		if (!isAsciiData)
		{
			read_binary_data(filedata, dimx, dimy, dimz, bytes, count, maxItem, voxels, minV, maxV, type);
		}
		else if (isAsciiData)
		{
			read_ascii_data(lines, dimx, dimy, dimz, bytes, voxels, minV, maxV);
		}
		float fdimx = dimx * spacingX, fdimy = dimy * spacingY, fdimz = dimz * spacingZ;
		float maxDim = std::max(std::max(fdimx, fdimy), fdimz);
		float xratio = fdimx / maxDim, yratio = fdimy / maxDim, zratio  = fdimz / maxDim;
		ui.openGLWidget->setData(voxels, { dimx, dimy, dimz }, { xratio, yratio, zratio }, minV, maxV);
		std::stringstream stream;
		stream << std::fixed << std::setprecision(0) << minV;
		ui.minLabel->setText(stream.str().c_str());
		stream = std::stringstream();
		stream << std::fixed << std::setprecision(0) << maxV;
		ui.maxLabel->setText(stream.str().c_str());
		ui.spinBox->setMinimum((int)minV);
		ui.spinBox->setMaximum((int)maxV);

	}

}

void read_ascii_data(QStringList &lines, int dimx, int dimy, int dimz, int bytes, std::vector<float> &voxels, double &minV, double &maxV)
{
	int x = 0, y = 0, z = 0;

	//reader.seek( file.size() - dimx * dimy * dimz * bytes);
	QString line;
	int index = 0, limit = dimx * dimy * dimz;
	int i = 10;
	while (i < lines.size()  && index < limit)
	{
		line = lines[i].simplified();
		//Remove empty line if any
		if (line.isEmpty())
		{
			continue;
			i++;
		}

		

		//QString CurrentTextLine = line;
		
		QStringList strs = line.split(" ");

		for (x = 0; x < dimx; x++)
		{
			if (strs.size() < dimx)
			{
				int dd = 4;
			}
			else if (strs.size() > dimx)
			{
				int dd = 4;
			}

			double density = strs[x].toDouble();
			double s = density;
			
			
			voxels[x + dimx * y + dimx * dimy * z] = (float)s;
			if (s < minV)
				minV = s;
			if (s > maxV)
				maxV = s;

			index++;
		}

		y++;
		if (y == dimy)
		{
			y = 0;
			z++;
		}
		i++;
	}
}

void read_binary_data(QByteArray &data, int dimx, int dimy, int dimz, int bytes, int &count,
	int maxItem, std::vector<float> &voxels, double &minV, double &maxV, QString  type)
{

	//QDataStream  reader(&file);
	//reader.skipRawData(file.size() - dimx * dimy * dimz * bytes + bytes);
	int i = data.size() - dimx * dimy * dimz * bytes ;
	//reader.setByteOrder(QDataStream::BigEndian);
	//Set the correct position
	int y = 0;
	int z = 0;
	int x = 0;
	
	int index = 0;
	while ((count != maxItem) && ( i < data.size()) != -1)
	{
		
		float tmp_data = 0;
		switch (bytes)
		{
		case 1:
			tmp_data = data[i];
			break;
		case 2:
			if (type == "unsigned_short")
			{
				char inverted[2] = {data[i], data[i+1] };
				unsigned short int tmp;
				std::memcpy(&tmp, inverted, sizeof tmp); // OK
				tmp_data = tmp;
				int u = 455;
			}
			else
			{
				char inverted[2] = { data[i], data[i + 1] };
				short tmp;
				std::memcpy(&tmp, inverted, sizeof tmp); // OK
				tmp_data = tmp;
			}
			break;
		case 4:
		{
			char inverted[4] = { data[i],data[i + 1], data[i + 2], data[i + 3] };
			std::memcpy(&tmp_data, inverted, sizeof tmp_data); // OK
			break;
		}
		default:
			break;
		}
		voxels[x + dimx * y + dimx * dimy * z] = tmp_data;
		index++;
		x++;
		i += bytes;
		if (x == dimx)
		{
			x = 0;
			y++;
			if (y == dimy)
			{
				y = 0;
				z++;
			}
		}
		//Set the min Max
		if (tmp_data < minV)
			minV = tmp_data;
		if (tmp_data > maxV)
			maxV = tmp_data;

		count++;
	}
}

void read_header(QStringList &in , int &dimx, int &dimy, int &dimz, bool &isAsciiData, int &bytes, float &spacingX,
	float &spacingY, float &spacingZ, QString &type)
{
	for (int i = 0; i < 10; i++)
	{
		QString line = in[i];
		//Remove comments
		while (line.contains("#", Qt::CaseInsensitive))
		{
			
			line = in[++i];
		}

		//Find the DIMENSIONS
		if (line.contains("DIMENSIONS"))
		{
			line = line.replace("DIMENSIONS", "");
			QStringList dim = line.split(QRegExp("\\s+"), QString::SkipEmptyParts);
			dimx = dim[0].toInt();
			dimy = dim[1].toInt();
			dimz = dim[2].toInt();
		}

		if (line.contains("BINARY"))
		{
			isAsciiData = false;
		}

		if (line.contains("LOOKUP_TABLE"))
		{
			break;
		}

		if (!isAsciiData && line.contains("SCALARS scalars"))
		{
			line = line.replace("SCALARS scalars ", "");
			type = line;
			if (line.compare("short 1", Qt::CaseInsensitive)  == 0 || line.compare("short", Qt::CaseInsensitive) == 0)
			{
				type = "short";
				bytes = 2;
			}
			else if (line.compare("unsigned_short", Qt::CaseInsensitive) == 0)
			{
				bytes = 2;
			}
			else if (line.compare("byte", Qt::CaseInsensitive) == 0)
			{
				bytes = 1;
			}
			else if (line.compare("float", Qt::CaseInsensitive) == 0)
			{
				bytes = 4;
			}
		}

		if (line.contains("SPACING"))
		{
			line = line.replace("SPACING", "");
			QStringList spacing = line.split(QRegExp("\\s+"), QString::SkipEmptyParts);
			spacingX = spacing[0].toFloat();
			spacingY = spacing[1].toFloat();
			spacingZ = spacing[2].toFloat();

			//Get the max
			float maxSpacing = 0;
			if (spacingX > spacingY)
				maxSpacing = spacingX;
			else
				maxSpacing = spacingY;

			if (spacingZ > maxSpacing)
				maxSpacing = spacingZ;

			//Normalize
			spacingX /= maxSpacing;
			spacingY /= maxSpacing;
			spacingZ /= maxSpacing;

		}
	}
}

void VolumeRaycaster::on_tfSlider_valueChanged(int pValue)
{
	float min = ui.tfSlider->minimum(), max = ui.tfSlider->maximum();
	ui.openGLWidget->setTF_offset((pValue - min) / (max - min));
}

void VolumeRaycaster::on_actionDensityMinus_triggered()
{
	ui.openGLWidget->decreaseDensity();
}

void VolumeRaycaster::on_actionDensityAdd_triggered()
{
	ui.openGLWidget->increaseDensity();
}

void VolumeRaycaster::on_actionLogScale_triggered()
{
	ui.openGLWidget->setLogScale(ui.actionLogScale->isChecked());
}

void VolumeRaycaster::on_spinBox_valueChanged(int pValue)
{
	ui.openGLWidget->setThreshold(ui.spinBox->value());
	ui.openGLWidget->setLogScale(ui.actionLogScale->isChecked());
}
