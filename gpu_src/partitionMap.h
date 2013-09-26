#ifndef PARTITION_MAP_H
#define PARTITION_MAP_H

#include <cv.h>
#include <stdlib.h>
#include <time.h>
#include <vector>


using namespace std;
using namespace cv;

enum gridType {
    OUTSIDE_ROOM = 0,
    WALL = 1,
    NOT_VISITED = 2,
    DECOMPOSED = 3
};

vector<GpuBlock*>* partitionMap (Mat&);
int countRoomSize(Mat&);
void initialRandomSeed(Mat&, int&, int&);
GpuBlock* getLargestRec(Mat&, int&, const int&, const int&);
int getMaxHeightRec(Mat&, const int&, const int&, int&, int&, int&, int&);
int getMaxWidthRec(Mat&, const int&, const int&, int&, int&, int&, int&);


vector<GpuBlock*>* partitionMap (Mat& model){
    vector<GpuBlock*>* blockList = new vector<GpuBlock*>;
    
    int roomSize = countRoomSize(model);
    int decomposedSize = 0;
    int i, j;
    while(decomposedSize != roomSize){
        initialRandomSeed(model, i, j);
        if(model.at<int>(i,j) != NOT_VISITED) continue;
        else 
        {
            //cout << i << "," << j << endl;
            GpuBlock* blk = getLargestRec(model, decomposedSize, i, j);
            blk->setSiblings(blockList);
            blockList->push_back(blk);
            //cout << decomposedSize << endl;
        }
    }
    /*
    for(int i = 0 ; i < blockList->size(); ++i){
        printf("size:%d, %d, %d, %d\n", 
               blockList->at(i)->upSibling.size(), 
               blockList->at(i)->downSibling.size(),
               blockList->at(i)->leftSibling.size(),
               blockList->at(i)->rightSibling.size()
              );
    }
    */
    return blockList;
}

int countRoomSize(Mat& M){
    int count = 0;
    for(int m = 0; m < M.rows; ++m)
        for(int n = 0; n < M.cols; ++n)
            if( M.at<int>(m,n) == NOT_VISITED) 
                count++;
    return count;
}

void initialRandomSeed(Mat& model, int& i, int& j){
    //srand ( time(NULL) );
    i = rand() % model.rows;
    j = rand() % model.cols;
    //cout << i << "," << j << endl;
    return;
}

GpuBlock* getLargestRec(Mat& M, int& decomposedSize, const int& i, const int& j){
    
    //First Rec: Max Height
    //Second Rec: Max Width
    //cout << i << "," << j << endl;
    int rec1RowLow, rec1RowHigh, rec1ColLow, rec1ColHigh;
    int rec2RowLow, rec2RowHigh, rec2ColLow, rec2ColHigh;
	int x_offset, y_offset, rows, cols;
    
    if( getMaxHeightRec(M, i, j, rec1RowLow, rec1RowHigh, rec1ColLow, rec1ColHigh) >= 
        getMaxWidthRec(M, i, j, rec2RowLow, rec2RowHigh, rec2ColLow, rec2ColHigh) )   {
        x_offset = rec1RowLow;
        y_offset = rec1ColLow;
        rows = rec1RowHigh - rec1RowLow + 1;
        cols = rec1ColHigh - rec1ColLow + 1;
        //printf("%d, %d, %d, %d\n", rec1RowLow, rec1RowHigh, rec1ColLow, rec1ColHigh);
        for(int m = rec1RowLow; m <= rec1RowHigh; ++m)
            for(int n = rec1ColLow; n <= rec1ColHigh; ++n)
                M.at<int>(m,n) = DECOMPOSED;
    }
    else{
        x_offset = rec2RowLow;
        y_offset = rec2ColLow;
        rows = rec2RowHigh - rec2RowLow + 1;
        cols = rec2ColHigh - rec2ColLow + 1;
        //printf("%d, %d, %d, %d\n", rec2RowLow, rec2RowHigh, rec2ColLow, rec2ColHigh);
        for(int m = rec2RowLow; m <= rec2RowHigh; ++m)
            for(int n = rec2ColLow; n <= rec2ColHigh; ++n)
                M.at<int>(m,n) = DECOMPOSED;
    }
    GpuBlock* block = new GpuBlock(x_offset, y_offset, rows, cols); //(x_offset, y_offset, rows, cols);
	//block->Init(x_offset, y_offset, rows, cols);
    //blockMaps.push_back(new GpuBlock(x_offset, y_offset, rows, cols));
    decomposedSize += rows * cols;
    return block;
};


int getMaxHeightRec(Mat& M, const int&i, const int&j, int& row_l,int& row_h,int& col_l, int& col_h)
{
    for(int m = i-1; m >= 0; --m)
        if( M.at<int>(m,j) != NOT_VISITED) {
            row_l = m + 1;
            break;
        }
    for(int m = i+1; m < M.rows; ++m)
        if( M.at<int>(m,j) != NOT_VISITED) {
            row_h = m - 1;
            break;
        }
    
    int height = row_h - row_l + 1;
    
    bool breakFlag;
    int n;
    breakFlag = false;
    for(n = j-1; n >= 0 && !breakFlag; --n){
        for(int m = row_l; m <= row_h; ++m){
            if( M.at<int>(m,n) != NOT_VISITED) { 
                col_l = n+1;
                breakFlag = true; 
                break;
            }
        }
    }

    
    breakFlag = false;
    for(n = j+1; n < M.cols && !breakFlag; ++n){
        for(int m = row_l; m <= row_h; ++m){
            if( M.at<int>(m,n) != NOT_VISITED) { 
                col_h = n-1;
                breakFlag = true; 
                break;
            }
        }
    }
   
    int width = col_h - col_l + 1;
    return (width * height);
};

int getMaxWidthRec(Mat& M, const int&i, const int&j, int& row_l,int& row_h,int& col_l, int& col_h)
{
    for(int n = j-1; n >= 0; --n)
        if( M.at<int>(i,n) != NOT_VISITED) {
            col_l = n + 1;
            break;
        }
    for(int n = j+1; n < M.cols; ++n)
        if( M.at<int>(i,n) != NOT_VISITED) {
            col_h = n - 1;
            break;
        }
    
    int width = col_h - col_l + 1;
    
    int m;
    bool breakFlag;
    breakFlag = false;
    for(m = i-1; m >= 0 && !breakFlag; --m){
        for(int n = col_l; n <= col_h; ++n){
            if( M.at<int>(m,n) != NOT_VISITED) { 
                row_l = m+1;  
                breakFlag = true;
                break;
            }
        }
    }
    
    
    breakFlag = false;
    for(m = i+1; m < M.rows && !breakFlag; ++m){
        for(int n = col_l; n <= col_h; ++n){
            if( M.at<int>(m,n) != NOT_VISITED) { 
                row_h = m-1;
                breakFlag = true; 
                break;
            }
        }
    }

   
    int height = row_h - row_l + 1;
    return (width * height);
};

#endif
