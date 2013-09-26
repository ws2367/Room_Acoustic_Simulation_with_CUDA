#include <QtGui/QApplication>

#include "paintwidget.h"
#include "musicsimulation.h"
#include "draglabel.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MusicSimulation w;
    w.show();

    return a.exec();
}
