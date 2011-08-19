#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cutil_inline.h>
#include <cv.h>
#include <cmath>
#include <time.h>
#include "highgui.h"

using namespace std;
using namespace cv;

#define PI 3.14159265358979323846
#define SOUND_SPEED 340.0
#define H 0.10675
#define DELTA_T (1.0/22050.0)
#define FNORM_FACTOR 100000000.0
//#define FNORM_FACTOR 1000.0
#define GRID_X_SIZE 16
#define GRID_Y_SIZE 16

#define CUDA_PI (3.141592653589793238462643383279502884197169399375105820974944f)
#define SIN_45 (0.70710678118654752440084436210485f)

#include "testTool.h"
#include "cuDCT.h"
#include "gpuBlock.h"
#include "partitionMap.h"
#include "utility.h"

const int duration = 44100;

int main(int argc, char** argv)
{ 
	assert(argc == 5);
	
	Mat model;
	int srcIdx_x, srcIdx_y;  //single source position
	
    //Parse the room model
    if(!getRoomModel(argv[1], model)){
        cout<<"Error: Opening room model failed.\n";
        return -1;
    }

    //Source point position
    stringstream tmpx(argv[2]);
    if( !(tmpx >> srcIdx_x) ) {cout << "Error: Source position x should be an integer.\n"; return -1;}
    stringstream tmpy(argv[3]);
    if( !(tmpy >> srcIdx_y) ) {cout << "Error: Source position y should be an integer.\n"; return -1;}
    srcIdx_x *=GRID_X_SIZE;
	srcIdx_y *=GRID_Y_SIZE;
    if(!isValidPosition(srcIdx_x, srcIdx_y, model)) {cout<<"Error: Invalid position of source.\n"; return -1;}
	cout<<srcIdx_x<<" "<<srcIdx_y<<endl;

    //initialize music data	
	//musicData = getMusicData(argv[2], musicLength);
    short* musicData;	
    musicData = ImpulseGenerate(22050, 2000, duration); //get gaussian impulse
	
	//partition the model to rectangles
	vector<GpuBlock*>* blockMaps;
	blockMaps = partitionMap(model);
    cout<<"Room model established and partitioned. Size ("<<model.rows<<","<<model.cols<<").\n";
    
	//reciever response
	int r_x = srcIdx_x;
	int r_y = srcIdx_y+2*GRID_Y_SIZE;
    if(!isValidPosition(r_x, r_y, model)) {cout<<"Error: Invalid position of reciever.\n"; return -1;}
	float* response  = new float[duration];
	int recvIdx = FindBlockIdx(blockMaps, r_x, r_y);
    if(recvIdx==-1){
        cout<<"Error: Invalid position of reciever.\n";
        return -1;
    }

    int subr_x = r_x - (*blockMaps)[recvIdx]->x_offset;
	int subr_y = r_y - (*blockMaps)[recvIdx]->y_offset;
    Mat totalMap = Mat::zeros(model.rows, model.cols, CV_32FC1);

  	printf("Number of blocks is %i. Enter simulation.\n", blockMaps->size());

    cufftComplex* h_p;

    //main simulation
	for(int i = 0; i < duration; ++i){  
		for(int j = 0; j < blockMaps->size(); ++j)
			(*blockMaps)[j]->UpdatePressure(i, musicData[i], srcIdx_x, srcIdx_y);
		for(int j = 0; j < blockMaps->size(); ++j)
			(*blockMaps)[j]->UpdateForce(musicData[i], srcIdx_x, srcIdx_y);
        if(i%100==0) cout << i << endl;

		//store response
		//response[i] = (*blockMaps)[recvIdx]->P.at<float>(subr_x, subr_y);
		//cout << response[i] <<endl;
		
        //show maps
        //if(i % 50 == 0) 
        showMap(blockMaps, totalMap, h_p, srcIdx_x, srcIdx_y, r_x, r_y);
        
        //showForce(blockMaps, totalMap, h_p, i);
        //waitKey(300);
        
	}

    

	//store response
	ofstream fout(argv[4], ios::binary);
	for(int i = 0; i < duration; ++i){
		if(response[i] > 32767.0 || response[i] < -32767.0)
			cout << response[i] <<endl;
		musicData[i] = (short)(response[i]);
	}
	
	fout.write((char*)musicData, duration*sizeof(short));
	fout.close();
    
	return 0;
}
