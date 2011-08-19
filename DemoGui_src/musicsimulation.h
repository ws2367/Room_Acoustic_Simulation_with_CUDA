#ifndef MUSICSIMULATION_H
#define MUSICSIMULATION_H

#include <QMainWindow>
#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QBitmap>
#include <QFile>
#include <QTextStream>
#include <QTextCursor>
#include <QPen>
//#include <filelock.hpp>

namespace Ui {
    class MusicSimulation;
}

class MusicSimulation : public QMainWindow
{
    Q_OBJECT

public:
    explicit MusicSimulation(QWidget *parent = 0);
    ~MusicSimulation();

    //paintWidget::paintEvent();

protected:
    void mouseMoveEvent(QMouseEvent *event);

private slots:
     void on_playButton_clicked();
     void on_loadButton_clicked();
     void on_stopButton_clicked();
     void on_originalButton_clicked();

private:
    Ui::MusicSimulation *ui;
    char* lockfile;
    char* positionfile;
};


#endif // MUSICSIMULATION_H
