#ifndef GPU_BLOCK_H
#define GPU_BLOCK_H
#include <cv.h>
#include <cmath>
#include <iostream>

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

    ~GpuBlock(){
        cudagpuBlockMemDealloc( plan, 
                                d_p_density, d_p_mode, d_t,
                                d_f_density, d_f_mode,
                                d_C1, d_C2,
                                d_DCT_coef_x , d_DCT_coef_y, 
                                d_IDCT_coef_x, d_IDCT_coef_y);
    }

	//initial funtions
	void Init(int global_x, int global_y, int rows, int cols)
	{
		x_offset = global_x;
		y_offset = global_y;
        rowSize = rows;
        colSize = cols; 

        cudagpuBlockMemAlloc(plan, &d_p_density, d_p_mode, &d_t,
                             &d_f_density, &d_f_mode, 
                             &d_C1, &d_C2, 
                             &d_DCT_coef_x, &d_DCT_coef_y, &d_IDCT_coef_x, &d_IDCT_coef_y, colSize, rowSize);
        
        cuda2dDCTSetCoef( d_C1, d_C2,
                          d_DCT_coef_x , d_DCT_coef_y, 
                          d_IDCT_coef_x, d_IDCT_coef_y, 
                          colSize, rowSize);
        //cout<<"----------------C1---------------"<<endl;
        //seeCoef(d_C1, colSize*rowSize);
        //waitKey(4000);
        //cout<<"----------------C2---------------"<<endl;
        //seeCoef(d_C2, colSize*rowSize);
        //waitKey(4000);
		//printf("GpuBlock(%i,%i) initialized.\n",x_offset, y_offset);
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
	void UpdatePressure(int timeIdx, short src, int src_x, int src_y)
	{
		int Idx = timeIdx % 3;
		int Idx_z1 = (Idx + 2) % 3;
		int Idx_z2 = (Idx + 1) % 3;

        //cufftComplex* h_f;
        
        /*
        UpdatePressure:

        DCT on f: d_f_density to d_f_mode
        updatePressure changes d_p_mode
        IDCT on p: d_p_mode to d_p_density
        
        UpdateForce:
        Update sound source: change d_f_density
        Update force: use d_p_density to update d_f_density*/
        
        cuda2dDCT(plan, d_f_mode, d_f_density, d_t, d_DCT_coef_x , d_DCT_coef_y , colSize, rowSize);
        
        //update rule  
        cudaUpdatePressure(d_f_mode, d_p_mode[Idx_z1], d_p_mode[Idx_z2], d_p_mode[Idx], d_C1, d_C2, colSize, rowSize);

        //h_f = (cufftComplex*) malloc(sizeof(cufftComplex)*colSize*rowSize);
        //getForce(d_p_mode[Idx_z1], h_f, colSize, rowSize,'1');
        //free(h_f);
  
        //h_f = (cufftComplex*) malloc(sizeof(cufftComplex)*colSize*rowSize);
        //getForce(d_p_mode[Idx], h_f, colSize, rowSize,'1');
        //free(h_f);

        cuda2dIDCT(plan, d_p_density, d_p_mode[Idx], d_t, d_IDCT_coef_x , d_IDCT_coef_y , colSize, rowSize);
        
		return;
	}

	void UpdateForce(short src, int src_x, int src_y)
	{
        //cout<<"update sound source.\n";
		//update sound source
		int x = src_x - x_offset;
		int y = src_y - y_offset;
		if(x >= 0 && y >= 0 && x < rowSize && y < colSize) //x here is vertical index, y is horizontal index
            cudaUpdateSoundSource(d_f_density, x, y, colSize, rowSize, src);

		//InterFace handling
		GpuBlock* blk;
        //cout<<"Interface handling.\n";

        //Change every if to max or min

		//Right
		for(int i = 0; i < rightSibling.size(); ++i){
            //printf("Right N.%i.\n",i);
			blk = rightSibling[i];
			int start = blk->x_offset - x_offset;
			if(start < 0) start = 0;
			int end = blk->x_offset + blk->rowSize - x_offset;
			if(end > rowSize) end = rowSize;
			int blkOffset = x_offset - blk->x_offset;
            
            cudaInterfaceHandling(d_p_density, blk->d_p_density, d_f_density, start, end, blkOffset, colSize, rowSize, blk->colSize, blk->rowSize, RIGHT);
		}

//        cout<<"Interface handling left.\n";		
		//Left
		for(int i = 0; i < leftSibling.size(); ++i){
            //printf("Left N.%i.\n",i);
			blk = leftSibling[i];
			int start = blk->x_offset - x_offset;
			if(start < 0) start = 0;
			int end = blk->x_offset + blk->rowSize - x_offset;
			if(end > rowSize) end = rowSize;
			int blkOffset = x_offset - blk->x_offset;

            cudaInterfaceHandling(d_p_density, blk->d_p_density, d_f_density, start, end, blkOffset, colSize, rowSize, blk->colSize, blk->rowSize, LEFT);
		}
//        cout<<"Interface handling up.\n";

		//UP	
		for(int i = 0; i < upSibling.size(); ++i){
            //printf("Up N.%i.\n",i);
			blk = upSibling[i];
			int start = blk->y_offset - y_offset;
			if(start < 0) start = 0;
			int end = blk->y_offset + blk->colSize - y_offset;
			if(end > colSize) end = colSize;
			int blkOffset = y_offset - blk->y_offset;

            cudaInterfaceHandling(d_p_density, blk->d_p_density, d_f_density, start, end, blkOffset, colSize, rowSize, blk->colSize, blk->rowSize, UP);
		}
//        cout<<"Interface handling down.\n";
		//DOWN
		for(int i = 0; i < downSibling.size(); ++i){
            //printf("Down N.%i.\n",i);
			blk = downSibling[i];
			int start = blk->y_offset - y_offset;
			if(start < 0) start = 0;
			int end = blk->y_offset + blk->colSize - y_offset;
			if(end > colSize) end = colSize;
			int blkOffset = y_offset - blk->y_offset;
            
            cudaInterfaceHandling(d_p_density, blk->d_p_density, d_f_density, start, end, blkOffset, colSize, rowSize, blk->colSize, blk->rowSize, DOWN);
		}

	}
    

	//DATA member

	//global offset
	int x_offset;
	int y_offset;
    
    int rowSize;
    int colSize;

	//sibling information
	vector<GpuBlock*> leftSibling;
	vector<GpuBlock*> rightSibling;
	vector<GpuBlock*> upSibling;
	vector<GpuBlock*> downSibling;

    //device pointer
	cufftHandle*     	plan;    
    cufftComplex*       d_p_density;
    cufftComplex*       d_p_mode[3];
	cufftComplex*		d_t;
    cufftComplex*       d_f_mode;
    cufftComplex*       d_f_density;

	//update constants
    cufftComplex*       d_C1;
    cufftComplex*       d_C2;
    cufftComplex*       d_DCT_coef_x;
    cufftComplex*       d_DCT_coef_y; 
    cufftComplex*       d_IDCT_coef_x;
    cufftComplex*       d_IDCT_coef_y;
};

int FindBlockIdx(vector<GpuBlock*>* blks,int g_x, int g_y)
{
	int Idx;
	for(Idx = 0; Idx < blks->size(); ++Idx){
		int sub_x = g_x - (*blks)[Idx]->x_offset;
		int sub_y = g_y - (*blks)[Idx]->y_offset;
		if(sub_x >= 0 && sub_x < (*blks)[Idx]->rowSize &&
           sub_y >= 0 && sub_y < (*blks)[Idx]->colSize )
			return Idx;
	}
	return -1;
};


#endif

