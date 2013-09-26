#include "paintwidget.h"

paintWidget::paintWidget(QWidget *parent)
    : QWidget(parent)
{
filename = "room_model3.txt";
}

paintWidget::~paintWidget()
{

}
void paintWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    //pen.setColor();
    painter.setPen(Qt::blue);
    painter.fillRect(16*18,16*3,16,16,Qt::SolidPattern);

    QPen pen;
    pen.setColor(Qt::blue);
    pen.setStyle(Qt::SolidLine);
    pen.setWidth(1);
    painter.setPen(pen);
    //painter.eraseRect(this->rect());

    int srcIdx_x, srcIdx_y;
    ifstream infile;
    infile.open(filename);
    if(!infile.is_open()) return;
    infile >> srcIdx_x >> srcIdx_y;
    char * buffer;
    buffer = new char[srcIdx_y];

    for(int i = 0; i < srcIdx_x; ++i){
       infile >> buffer;
       for(int j = 0; j < srcIdx_y; ++j){
           //this->setToolTip(QString(buffer[j]));
           if( buffer[j]== '1'){
               //QPointF points[4]={QPointF(16*j, 16*i), QPointF(16*j+15, 16*i),
               //                   QPointF(16*j+15, 16*i+15), QPointF(16*j, 16*i+15)};
               //painter.drawPolygon(points,4);
               for(int k=0; k<9; k++)
                    painter.drawRect(16*j+k,16*i+k,16-2*k,16-2*k);
           }
        }
    }
    QPen pen2;

    infile.close();
    painter.end();
    return;
}

void paintWidget::paint(char* file){
    filename = file;
    this->setToolTip(QString(filename));
    this->update();
    //this->repaint();
    return;
}
