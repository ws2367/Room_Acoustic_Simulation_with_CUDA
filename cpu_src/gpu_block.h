#ifndef GPU_BLOCK_H
#define GPU_BLOCK_H
#include <cv.h>
#include <cmath>

#define PI 3.14159265358979323846
#define SOUND_SPEED 340.0
#define H 0.10675
#define DELTA_T (1.0/22050.0)
#define FNORM_FACTOR 100000000.0

using namespace std;
using namespace cv;

class GpuBlock
{
public:
	GpuBlock(){}
	GpuBlock(int x, int y, int r, int c) {
        Init(x, y, r, c);
    }
    GpuBlock(const GpuBlock& b) {
        Init(b.x_offset, b.y_offset, b.rowSize, b.colSize);
    }
	//initial funtions
	void Init(int global_x, int global_y, int rows, int cols)
	{
		x_offset = global_x;
		y_offset = global_y;
        rowSize = rows;
        colSize = cols; 
    
		//initial matrix varibles
		P = Mat::zeros(rows, cols, CV_64FC1);
		F = Mat::zeros(rows, cols, CV_64FC1);
		F_dct = Mat::zeros(rows, cols, CV_64FC1);
		M[0] = Mat::zeros(rows, cols, CV_64FC1);
		M[1] = Mat::zeros(rows, cols, CV_64FC1);
		M[2] = Mat::zeros(rows, cols, CV_64FC1);

		//initial update constants
		C1 = Mat::zeros(rows, cols, CV_64FC1);
		C2 = Mat::zeros(rows, cols, CV_64FC1);
		double lx = rows * H;
		double ly = cols * H;
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				if(i==0 && j == 0){              // DC special case
					C1.at<double>(0,0) = 1.0;
					C2.at<double>(0,0) = 0.0;
					continue;
				}
				double omega = SOUND_SPEED * PI *
                               sqrt((double)i*i/lx/lx + (double)j*j/ly/ly);
				double cosomega = cos(omega*DELTA_T);
				C1.at<double>(i,j) = 2.0 * cosomega;
				C2.at<double>(i,j) = 2.0 * (1-cosomega) / omega /omega; 
			}
		}
		return;
	}
    
    void setSiblings(vector<GpuBlock*>* blockList ){
        int x_high = x_offset + rowSize - 1 ;
        int y_high = y_offset + colSize - 1;
        
        //printf("current block domain:%d, %d, %d, %d\n", x_offset, x_high,y_offset,y_high);
        int x_offset_t, y_offset_t, x_high_t, y_high_t;
        for(int i = 0; i < blockList->size(); ++i){
            x_offset_t = blockList->at(i)->x_offset;
            y_offset_t = blockList->at(i)->y_offset;
            x_high_t = x_offset_t + blockList->at(i)->rowSize - 1;
            y_high_t = y_offset_t + blockList->at(i)->colSize - 1;
            //printf("blockList block domain:%d, %d, %d, %d\n", x_offset_t, x_high_t,y_offset_t, y_high_t);
            //current block is at the up of ith block
            if( ((x_high+1) == x_offset_t) && y_high >= y_offset_t && y_offset <= y_high_t ){
                downSibling.push_back(blockList->at(i));
                blockList->at(i)->upSibling.push_back(this);
            }
            else if(((x_offset-1) == x_high_t) && y_high >= y_offset_t && y_offset <= y_high_t){
                upSibling.push_back(blockList->at(i));
                blockList->at(i)->downSibling.push_back(this);
            }
            else if(((y_high+1) == y_offset_t) && x_high >= x_offset_t && x_offset <= x_high_t){
                rightSibling.push_back(blockList->at(i));
                blockList->at(i)->leftSibling.push_back(this);
            }
            else if(((y_offset-1) == y_high_t) && x_high >= x_offset_t && x_offset <= x_high_t){
                leftSibling.push_back(blockList->at(i));
                blockList->at(i)->rightSibling.push_back(this);
            }
            
        }
    }
    
	//update function
	void UpdatePressure(int timeIdx)
	{
		int Idx = timeIdx % 3;
		int Idx_z1 = (Idx + 2) % 3;
		int Idx_z2 = (Idx + 1) % 3;

		dct(F, F_dct);
		multiply(M[Idx_z1], C1, temp1);
		multiply(F_dct, C2, temp2);
		M[Idx] = temp1 - M[Idx_z2] + temp2;
		idct(M[Idx], P);
		
		return;
	}

	void UpdateForce(short src, int src_x, int src_y)
	{
		//update sound source
		int x = src_x - x_offset;
		int y = src_y - y_offset;
		if(x >= 0 && y >= 0 && x < F.rows && y < F.cols)
			F.at<double>(x, y) = src * FNORM_FACTOR;

		//InterFace handling
		GpuBlock* blk;

		//Right
		for(int i = 0; i < rightSibling.size(); ++i){
			blk = rightSibling[i];
			int start = blk->x_offset - x_offset;
			if(start < 0) start = 0;
			int end = blk->x_offset + blk->P.rows - x_offset;
			if(end > P.rows) end = P.rows;
			int blkOffset = x_offset - blk->x_offset;
			for(int j = start; j < end; ++j){
				double Si = - 2.0 * P.at<double>(j, P.cols - 3)
				            + 27.0 * P.at<double>(j, P.cols - 2)
				            - 270.0 * P.at<double>(j, P.cols -1)
				            + 270.0 * blk->P.at<double>(j + blkOffset, 0)
				            - 27.0 * blk->P.at<double>(j + blkOffset, 1)
				            + 2.0 * blk->P.at<double>(j + blkOffset, 2);
				F.at<double>(j, P.cols - 1) = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
			}
		}
		
		//Left
		for(int i = 0; i < leftSibling.size(); ++i){
			blk = leftSibling[i];
			int start = blk->x_offset - x_offset;
			if(start < 0) start = 0;
			int end = blk->x_offset + blk->P.rows - x_offset;
			if(end > P.rows) end = P.rows;
			int blkOffset = x_offset - blk->x_offset;
			for(int j = start; j < end; ++j){
				double Si = - 2.0 * P.at<double>(j, 2)
							+ 27.0 * P.at<double>(j, 1)
							- 270.0 * P.at<double>(j, 0)
							+ 270.0 * blk->P.at<double>(j + blkOffset, blk->P.cols - 1)
							- 27.0 * blk->P.at<double>(j + blkOffset, blk->P.cols - 2)
							+ 2.0 * blk->P.at<double>(j + blkOffset, blk->P.cols -3);
				//Si = Si * (SOUND_SPEED * SOUND_SPEED);
				F.at<double>(j, 0) = Si * SOUND_SPEED * SOUND_SPEED  / (180.0 * H * H);
			}
		}
	
		//UP	
		for(int i = 0; i < upSibling.size(); ++i){
			blk = upSibling[i];
			int start = blk->y_offset - y_offset;
			if(start < 0) start = 0;
			int end = blk->y_offset + blk->P.cols - y_offset;
			if(end > P.cols) end = P.cols;
			int blkOffset = y_offset - blk->y_offset;
			for(int j = start; j < end; ++j){
				double Si = - 2.0 * P.at<double>(2, j)
				            + 27.0 * P.at<double>(1, j)
				            - 270.0 * P.at<double>(0, j)
				            + 270.0 * blk->P.at<double>(blk->P.rows - 1, j + blkOffset)
				            - 27.0 * blk->P.at<double>(blk->P.rows - 2, j + blkOffset)
				            + 2.0 * blk->P.at<double>(blk->P.rows - 3, j + blkOffset);
				F.at<double>(0, j) = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
			}
		}

		//DOWN
		for(int i = 0; i < downSibling.size(); ++i){
			blk = downSibling[i];
			int start = blk->y_offset - y_offset;
			if(start < 0) start = 0;
			int end = blk->y_offset + blk->P.cols - y_offset;
			if(end > P.cols) end = P.cols;
			int blkOffset = y_offset - blk->y_offset;
			for(int j = start; j < end; ++j){
				double Si = - 2.0 * P.at<double>(P.rows - 3, j)
				            + 27.0 * P.at<double>(P.rows - 2, j)
				            - 270.0 * P.at<double>(P.rows - 1, j)
				            + 270.0 * blk->P.at<double>(0, j + blkOffset)
				            - 27.0 * blk->P.at<double>(1, j + blkOffset)
				            + 2.0 * blk->P.at<double>(2, j + blkOffset);
				F.at<double>(P.rows - 1, j) = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
			}
		}


	}
    

	//DATA member

	//global offset
	int x_offset;
	int y_offset;
    
    int rowSize;
    int colSize;
	
	//update constants
	Mat C1;
	Mat C2;
	//update temp var
	Mat temp1, temp2, temp3;
	//block informtion
	Mat P;
	Mat M[3];
	Mat F;
	Mat F_dct;

	//sibling information
	vector<GpuBlock*> leftSibling;
	vector<GpuBlock*> rightSibling;
	vector<GpuBlock*> upSibling;
	vector<GpuBlock*> downSibling;
};


int FindBlockIdx(vector<GpuBlock*>* blks,int g_x, int g_y)
{
	int Idx;
	for(Idx = 0; Idx < blks->size(); ++Idx){
		int sub_x = g_x - (*blks)[Idx]->x_offset;
		int sub_y = g_y - (*blks)[Idx]->y_offset;
		if(sub_x >= 0 && sub_x < (*blks)[Idx]->P.rows &&
           sub_y >= 0 && sub_y < (*blks)[Idx]->P.cols )
			return Idx;
	}
	return -1;
};


#endif
