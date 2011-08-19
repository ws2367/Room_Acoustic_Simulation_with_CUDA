#include "musicsimulation.h"
#include "ui_musicsimulation.h"


bool checkFilelock(const char* lockfile)
{
        ifstream testlock(lockfile);
        if(testlock){

#ifdef DEBUG
                printf("Filelock \"%s\" already exist! \n", lockfile);
#endif
                return false;
        }else{

#ifdef DEBUG
                printf("File lock \"%s\" check ok \n", lockfile);
#endif
                return true;
        }

}

void getFilelock(const char* lockfile)
{
        ofstream lock(lockfile);
        #ifdef DEBUG
        printf("get file lock %s \n", lockfile);
        #endif
}

void releaseFilelock(const char* lockfile)
{
        remove(lockfile);
        #ifdef DEBUG
        printf("release file lock %s \n", lockfile);
        #endif
}

MusicSimulation::MusicSimulation(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MusicSimulation)
{
    ui->setupUi(this);
    lockfile="lockfile";
    positionfile="positionfile";
    remove(positionfile);
}

MusicSimulation::~MusicSimulation()
{
    delete ui;
}

void MusicSimulation::mouseMoveEvent(QMouseEvent *event) {
    QString msg;
    int x,y;
    x=(event->x()-10)>0?(event->x()-10):0;
    y=(event->y()-70)>0?(event->y()-70):0;
    msg.sprintf("Move: (%d, %d)", x, y);
    ui->stateLabel->setText(msg);

    while(!checkFilelock(lockfile)){}
    getFilelock(lockfile);
    ifstream infile(positionfile);
    int tmp;
    if(infile.is_open()){
        infile>>tmp;
        infile>>tmp;
        infile>>tmp;
        infile.close();
        remove(positionfile);
    }
    else
        tmp=0;
    ofstream outfile(positionfile,ios::trunc);
    //int x,y;
    //x=(event->x()-10)>0?(event->x()-10):0;
    //y=(event->y()-70)>0?(event->y()-70):0;
    outfile<<x<<" "<<y<<" "<<tmp;
    outfile.close();
    releaseFilelock(lockfile);
    //ui->widget->paintPt(event->x()-10, event->y()-70);
    return;
}

void MusicSimulation::on_playButton_clicked()
{
    while(!checkFilelock(lockfile)){}
    getFilelock(lockfile);
    ifstream infile(positionfile);
    int x,y;
    infile>>x;
    infile>>y;
    infile.close();
    remove(positionfile);
    ofstream outfile(positionfile,ios::trunc);
    outfile<<x<<" "<<y<<" "<<1;
    outfile.close();
    releaseFilelock(lockfile);
}

void MusicSimulation::on_loadButton_clicked()
{
    //char *file=ui->comboBox->currentText().toAscii().data();
    //this->update();
    //ui->widget->paint(file);
}

void MusicSimulation::on_stopButton_clicked()
{
    while(!checkFilelock(lockfile)){}
    getFilelock(lockfile);
    ifstream infile(positionfile);
    int x,y;
    infile>>x;
    infile>>y;
    infile.close();
    remove(positionfile);
    ofstream outfile(positionfile,ios::trunc);
    outfile<<x<<" "<<y<<" "<<0;
    outfile.close();
    releaseFilelock(lockfile);
}


void MusicSimulation::on_originalButton_clicked()
{
    while(!checkFilelock(lockfile)){}
    getFilelock(lockfile);
    ifstream infile(positionfile);
    int x,y;
    infile>>x;
    infile>>y;
    infile.close();
    remove(positionfile);
    ofstream outfile(positionfile,ios::trunc);
    outfile<<x<<" "<<y<<" "<<2;
    outfile.close();
    releaseFilelock(lockfile);
}
