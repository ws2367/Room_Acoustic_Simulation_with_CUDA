#include <cv.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "gpu_block.h"
#include "gpu_config.h"
#include "partitionMap.h"
#include "highgui.h"
#include "impulseGenerate.h"

using namespace cv;
using namespace std;

const int duration = 44100;

bool isValidPosition(const int x, const int y, Mat& model){
    return (model.at<int>(x,y)==DECOMPOSED)||(model.at<int>(x,y)==NOT_VISITED);
}

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
	
	// get the whole music (assume 16-bit pcm raw data)
	
    int musicLength;    //data length
	short* musicData;
	musicData = ImpulseGenerate(22050, 2000, duration); //get gaussian impulse
	
	//partition the model to rectangles
	vector<GpuBlock*>* blockMaps;
	blockMaps = partitionMap(model);
    cout<<"Room model established and partitioned. Size ("<<model.rows<<","<<model.cols<<").\n";


	Mat totalMap = Mat::zeros(model.rows, model.cols, CV_64FC1);
        totalMap = totalMap + 128;
	//reciever response
	int sub_rows = model.rows/16;
	int sub_cols = model.cols/16;
	short** IR = new short*[sub_rows * sub_cols];
	int* recvIdx = new int[sub_rows * sub_cols];
	int* subr_x = new int[sub_rows * sub_cols];
	int* subr_y = new int[sub_rows * sub_cols];

	for(int i = 0; i < sub_rows; ++i){
		for(int j = 0; j < sub_cols; ++j){
			IR[i*sub_cols+j] = new short[duration];
			if(!IR[i*sub_cols + j]){
				cout << "memory not enough!" << endl; 
				return -1;
			}
			
			recvIdx[i*sub_cols+j] = FindBlockIdx(blockMaps, i*16, j*16);
			if(recvIdx[i*sub_cols]>=0){
				subr_x[i*sub_cols+j] = i*16 - (*blockMaps)[recvIdx[i*sub_cols+j]]->x_offset;
				subr_y[i*sub_cols+j] = j*16 - (*blockMaps)[recvIdx[i*sub_cols+j]]->y_offset;
			}
			//cout << i << " " << j << " " << recvIdx[i*sub_cols+j];
		}
	}


	for(int i = 0; i < duration; ++i){  //main simulation

		for(int j = 0; j < blockMaps->size(); ++j)
			(*blockMaps)[j]->UpdatePressure(i);
		for(int j = 0; j < blockMaps->size(); ++j)
			(*blockMaps)[j]->UpdateForce(musicData[i], srcIdx_x, srcIdx_y);	
		cout << i << "\n";

		//store response
		//response[i] = (*blockMaps)[recvIdx]->P.at<double>(subr_x, subr_y);
		//cout << response[i] <<endl;
		double res;
		for(int Idx = 0; Idx < sub_rows * sub_cols; ++Idx){
			if(recvIdx[Idx] >= 0){
				res = (*blockMaps)[recvIdx[Idx]]->P.at<double>(subr_x[Idx], subr_y[Idx]);
				if(res > 32767.0)
					res = 32767.0;
				if(res < -32767.0)
					res = -32767.0;
				IR[Idx][i] = (short)res;
			}
		}

		//show maps
		Mat img;
		if(i % 50 == 0){
			for(int k = 0; k < blockMaps->size(); ++k){
				int off_x = (*blockMaps)[k]->x_offset;
				int off_y = (*blockMaps)[k]->y_offset;
				for(int x = 0; x < (*blockMaps)[k]->P.rows; ++x)
					for(int y = 0; y < (*blockMaps)[k]->P.cols; ++y)
						totalMap.at<double>(x+off_x, y+off_y) = (*blockMaps)[k]->P.at<double>(x,y);
			}
		}
		Mat(totalMap).convertTo(img, CV_8UC1);
		imshow("P", img);
		waitKey(5);
	}


	//store response
	ofstream fout(argv[4], ios::binary);
	fout.write((char*)&duration, sizeof(int));
	fout.write((char*)&sub_rows, sizeof(int));
	fout.write((char*)&sub_cols, sizeof(int));
	for(int i = 0; i < sub_rows*sub_cols; ++i)
		fout.write((char*)IR[i], duration*sizeof(short));	

	//fout.write((char*)musicData, duration*sizeof(short));
	fout.close();
    
	return 0;
}
