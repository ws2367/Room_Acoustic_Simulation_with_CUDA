#include "draglabel.h"

#include <QApplication>
#include <QPoint>
#include <QMouseEvent>
#include <QMimeData>
#include <QDrag>
#include <QIcon>

dragLabel::dragLabel(QLabel *parent) : QLabel(parent) {
    imgname ="ear.gif";
    readImage(QString(imgname));
    setAcceptDrops(true);
}

dragLabel::~dragLabel(){
}

void dragLabel::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        startPoint = event->pos();
    }
    QLabel::mousePressEvent(event);
}

void dragLabel::mouseMoveEvent(QMouseEvent *event) {
    if (event->buttons() & Qt::LeftButton) {
        if ((event->pos() - startPoint).manhattanLength()
                >= QApplication::startDragDistance()) {
            execDrag();
        }
    }
    QLabel::mouseMoveEvent(event);
}

void dragLabel::execDrag() {
    //QLabelItem *item = currentItem();
        //if (item) {
            QMimeData *mimeData = new QMimeData;
            //mimeData->setText(item->text());
            //mimeData->setImageData(item->icon());
            QDrag *drag = new QDrag(this);
            drag->setMimeData(mimeData);
            //drag->setPixmap(item->icon().pixmap(QSize(22, 22)));
            if (drag->exec(Qt::MoveAction) == Qt::MoveAction) {
                //delete item;
           }
}

void dragLabel::dragEnterEvent(QDragEnterEvent *event) {
    dragLabel *source =
            qobject_cast<dragLabel *>(event->source());
    if (source && source != this) {
        event->setDropAction(Qt::MoveAction);
        event->accept();
    }
}

void dragLabel::dragMoveEvent(QDragMoveEvent *event) {}

void dragLabel::dropEvent(QDropEvent *event) {
    /*dragLabel *source =
            qobject_cast<dragLabel *>(event->source());
    if (source && source != this) {
        QIcon icon = event->mimeData()->imageData().value<QIcon>();
        QString text = event->mimeData()->text();
        addItem(new QdragLabelItem(icon, text));
*/
        event->setDropAction(Qt::MoveAction);
        event->accept();
  //  }
}

void dragLabel::readImage(const QString &fileName){
    QPixmap pixmap(fileName);
    this->setPixmap(pixmap);
    this->resize(pixmap.width(), pixmap.height());
}
