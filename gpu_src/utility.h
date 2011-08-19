#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cv.h>
#include <cmath>


using namespace cv;
using namespace std;

bool getRoomModel(char* filename, Mat& model)
{
    ifstream infile;
    infile.open(filename);
    if(!infile.is_open()) return false;
    int nrows, ncols;
    infile >> nrows >> ncols ;
    model = Mat(nrows * GRID_X_SIZE, ncols * GRID_Y_SIZE, CV_32SC1);
    char * buffer;
    buffer = new char [ncols];
    for(int i = 0; i < nrows; ++i){
        infile >> buffer;
        for(int j = 0; j < ncols; ++j)
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

bool isValidPosition(const int x, const int y, Mat& model){
    return (model.at<int>(x,y)==DECOMPOSED)||(model.at<int>(x,y)==NOT_VISITED);
}

short* ImpulseGenerate(double sampleRate, double highestRate, int duration)
{
	double lo = 1.0/(PI*highestRate);
	short* impulse = new short[duration];
	double t_step = 1.0/sampleRate;

	for(int i = 0; i < duration; ++i){
		double k = t_step*i - 5*lo;
		impulse[i] = (short)(10000.0*exp(-k*k/lo/lo));
	}

	return impulse;
}

void getPressure(cufftComplex*& d_p_density, Mat& P, cufftComplex* h_p, const int colSize, const int rowSize){
    if(h_p==NULL) cout<<"Error: Memory allocation failed in reading pressure field on GPU memory.\n";
    cutilSafeCall(cudaMemcpy(h_p, d_p_density, sizeof(cufftComplex)*colSize*rowSize, cudaMemcpyDeviceToHost));
/*
    double max=0, min=0;
    for(int i=0;i<colSize*rowSize; i++){
        if(max<h_p[i].x) max=h_p[i].x;
        if(min>h_p[i].x) min=h_p[i].x;
    }
    //printf("h_p max:%e, min:%e.\n",max,min);    

    for(int i=0; i<colSize; i++)
        for(int j=0; j<rowSize; j++)
            h_p_trans[i*rowSize+j]=h_p[i+j*colSize];
*/    
    complex2Float( (float*)P.data, h_p, colSize, rowSize);
    
    return;
}

void showMap(vector<GpuBlock*>*& blockMaps, Mat& totalMap, cufftComplex* h_p, 
             const int s_x, const int s_y, const int r_x, const int r_y){
    Mat img;
    Mat P;
	for(int k = 0; k < blockMaps->size(); ++k){    

        P = Mat::zeros((*blockMaps)[k]->rowSize, (*blockMaps)[k]->colSize, CV_32FC1);        
        h_p = (cufftComplex*) malloc(sizeof(cufftComplex)*(*blockMaps)[k]->colSize*(*blockMaps)[k]->rowSize);
        getPressure((*blockMaps)[k]->d_p_density, P, h_p, (*blockMaps)[k]->colSize, (*blockMaps)[k]->rowSize); 

		int off_x = (*blockMaps)[k]->x_offset;
		int off_y = (*blockMaps)[k]->y_offset;        
        //float max=0.0, min=0.0;
		for(int x = 0; x < (*blockMaps)[k]->rowSize; ++x)
			for(int y = 0; y < (*blockMaps)[k]->colSize; ++y){
				totalMap.at<float>(x+off_x, y+off_y) = P.at<float>(x,y);
                //if(P.at<float>(x,y)>max) max = P.at<float>(x,y);
                //if(P.at<float>(x,y)<min) min = P.at<float>(x,y);
        }
        P.release();
        free(h_p);
        //if(h_p==NULL) cout<<"free succeeded.\n";
        //cout<<"P: "<<max<<"-"<<min<<endl;
	}
    //perform as saturate_cast<>
	Mat(totalMap).convertTo(img, CV_8UC1);
    
    img.at<char>(s_x,s_y)=char(255);
    img.at<char>(r_x,r_y)=char(255);
	imshow("P", img);
	waitKey(5);
	
    return;
}

void showForce(vector<GpuBlock*>*& blockMaps, Mat& totalMap, cufftComplex* h_p, int timeIdx){
    Mat img;
    Mat P;
    int Idx = timeIdx % 3;

	for(int k = 0; k < blockMaps->size(); ++k){    
        P = Mat::zeros((*blockMaps)[k]->rowSize, (*blockMaps)[k]->colSize, CV_32FC1);        
        h_p = (cufftComplex*) malloc(sizeof(cufftComplex)*(*blockMaps)[k]->colSize*(*blockMaps)[k]->rowSize);
		int off_x = (*blockMaps)[k]->x_offset;
		int off_y = (*blockMaps)[k]->y_offset;
        getPressure((*blockMaps)[k]->d_p_mode[Idx], P, h_p, (*blockMaps)[k]->colSize, (*blockMaps)[k]->rowSize); 
//        float max=0.0, min=0.0;
		for(int x = 0; x < (*blockMaps)[k]->rowSize; ++x)
			for(int y = 0; y < (*blockMaps)[k]->colSize; ++y){
				totalMap.at<float>(x+off_x, y+off_y) = P.at<float>(x,y);
//                if(P.at<float>(x,y)>max) max = P.at<float>(x,y);
//                if(P.at<float>(x,y)<min) min = P.at<float>(x,y);
        }
        P.release();
        free(h_p);
//        if(h_p==NULL) cout<<"free succeeded.\n";
//        cout<<"P_mode: "<<max<<"-"<<min<<endl;
	}
    //perform as saturate_cast<>
	Mat(totalMap).convertTo(img, CV_8UC1);
    
	imshow("P_mode", img);
	waitKey(5);
	
    return;
}


#endif
