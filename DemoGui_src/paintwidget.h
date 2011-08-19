#ifndef PAINTWIDGET_H
#define PAINTWIDGET_H

#include <iostream>
#include <QtGui/QWidget>
#include <QPainter>
#include <QFile>
#include <fstream>

using namespace std;

class paintWidget : public QWidget
{
    Q_OBJECT

public:
    paintWidget(QWidget *parent = 0);
    ~paintWidget();

    void paint(char* file);

protected:
    void paintEvent(QPaintEvent *);

private:
    char* filename;
};

#endif // PAINTWIDGET_H
