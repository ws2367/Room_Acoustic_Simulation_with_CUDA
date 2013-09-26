#ifndef DRAGLABEL_H
#define DRAGLABEL_H

#include <QLabel>

class QDragEnterEvent;
class QDropEvent;

class dragLabel : public QLabel {
   Q_OBJECT

public:
    dragLabel(QLabel *parent = 0);
    ~dragLabel();

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void dragEnterEvent(QDragEnterEvent *event);
    void dragMoveEvent(QDragMoveEvent *event);
    void dropEvent(QDropEvent *event);

private:
    void readImage(const QString &fileName);
    char* imgname;
    void execDrag();
    QPoint startPoint;
};

#endif // DRAGLABEL_H
