#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cv.h>

#define GRID_X_SIZE 16
#define GRID_Y_SIZE 16
using namespace cv;
using namespace std;

void getConfig(char* filename, double& delta_t, double& delta_l)
{



};


bool getRoomModel(char* filename, Mat& model)
{
	int srcIdx_x, srcIdx_y;
    ifstream infile;
    infile.open(filename);
	if(!infile) return false;
    infile >> srcIdx_x >> srcIdx_y ;
    model = Mat(srcIdx_x * GRID_X_SIZE, srcIdx_y * GRID_Y_SIZE, CV_32SC1);
    char * buffer;
    buffer = new char [srcIdx_y];
    
    for(int i = 0; i < srcIdx_x; ++i){
        infile >> buffer;
        for(int j = 0; j < srcIdx_y; ++j)
            for(int m = 0; m < GRID_X_SIZE; ++m)
                for(int n = 0; n < GRID_Y_SIZE; ++n)
                    model.at<int>(i*GRID_X_SIZE+m,j*GRID_Y_SIZE+n) = buffer[j] - '0';
                
    }  
    infile.close();
	return true;
};


short* getMusicData(char* filename, int& musicLength)
{
	ifstream is(filename, ios::binary);
	// get length of file:
	is.seekg (0, ios::end);
	musicLength = is.tellg()/2;
	is.seekg (0, ios::beg);

	// allocate memory:
	short* buffer = new short [musicLength];
	is.read((char*)buffer,musicLength*sizeof(short));

	is.close();
	return buffer;
};

#endif
