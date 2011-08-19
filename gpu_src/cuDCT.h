#ifndef CUDCT_H
#define CUDCT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

#include <cuda.h>
#include <cufft.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

#define BLOCK_NUM 256
#define THREAD_NUM 256

enum SiblingType {
    RIGHT=1,
    LEFT=2,
    UP=3,
    DOWN=4
};

static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
void float2Complex(cufftComplex*, float*, int, int);
void complex2Float(float*, cufftComplex*, int, int);

void cudagpuBlockMemAlloc( cufftHandle* &, 
                           cufftComplex **, cufftComplex *[3], 
                           cufftComplex **, 
                           cufftComplex **, cufftComplex **, 
                           cufftComplex **, cufftComplex **, 
                           cufftComplex **, cufftComplex **, 
                           cufftComplex **, cufftComplex **, 
                           int, int);

void cudagpuBlockMemDealloc( cufftHandle* &, 
                             cufftComplex *, cufftComplex *[3], 
                             cufftComplex *, 
                             cufftComplex *, cufftComplex *, 
                             cufftComplex *, cufftComplex *,
                             cufftComplex *, cufftComplex *, 
                             cufftComplex *, cufftComplex *);

void cuda2dDCTSetCoef ( cufftComplex*, cufftComplex*, 
                        cufftComplex*, cufftComplex*, 
                        cufftComplex*, cufftComplex*, 
                        int, int);

void cuda2dDCT  (cufftHandle* &, cufftComplex*, cufftComplex*, 
                 cufftComplex*, cufftComplex*, cufftComplex*, int, int);
void cuda2dIDCT (cufftHandle* &, cufftComplex*, cufftComplex*, 
                 cufftComplex*, cufftComplex*, cufftComplex*, int, int);
void cudaUpdatePressure (cufftComplex*, cufftComplex*, cufftComplex*, cufftComplex*, cufftComplex*, cufftComplex*, int, int);
void cudaInterfaceHandling(cufftComplex*, cufftComplex*, cufftComplex*, const int, const int, const int, const int, const int, const int, const int, SiblingType);
void cudaUpdateSoundSource(cufftComplex*, const int, const int, const int, const float);

static __global__ void cuUpdatePressure(cufftComplex*, cufftComplex*, cufftComplex*, 
                                        cufftComplex*, cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuC12SetCoef(cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuDCTSetCoef(cufftComplex*, const int);
static __global__ void cuIDCTSetCoef(cufftComplex*, const int);
static __global__ void cuDCTSwap(cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuIDCTSwap(cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuDCTCoefMul(cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuIDCTCoefMul(cufftComplex*, cufftComplex*, const int , const int);
static __global__ void cuTrans (cufftComplex*, cufftComplex*, const int, const int);
static __global__ void cuPassForceToRight(cufftComplex*, cufftComplex*, cufftComplex*, 
                                          const int, const int, const int, const int, const int, const int, const int);
static __global__ void cuPassForceToLeft(cufftComplex*, cufftComplex*, cufftComplex*, 
                                         const int, const int, const int, const int, const int, const int, const int);
static __global__ void cuPassForceToUp(cufftComplex*, cufftComplex*, cufftComplex*, 
                                       const int, const int, const int, const int, const int, const int, const int);
static __global__ void cuPassForceToDown(cufftComplex*, cufftComplex*, cufftComplex*, 
                                         const int, const int, const int, const int, const int, const int, const int);

static __global__ void cuUpdateSoundSource(cufftComplex*, const int, const int, const int, const int, const float);
static __global__ void cuSetZero(cufftComplex*, const int);

void float2Complex(cufftComplex *data_complex, float *data, int nx, int ny){
    for (int i = 0; i < nx*ny; ++i) {
        data_complex[i].x = data[i];
        data_complex[i].y = 0;
        //printf("(%2f, %2f) ", data_complex[i].x, data_complex[i].y);
    }
};

void complex2Float(float* data, cufftComplex* data_complex, int nx, int ny){
    for (int i = 0; i < nx*ny; ++i) {
        data[i] = data_complex[i].x;
        //printf("%2f ", data_[i]);
    }
};


void cudagpuBlockMemDealloc( cufftHandle* &plan, 
                             cufftComplex *d_p_density,   cufftComplex *d_p_mode[3], cufftComplex *d_t,          
                             cufftComplex *d_f_density,   cufftComplex *d_f_mode,
                             cufftComplex *d_C1,          cufftComplex *d_C2,
                             cufftComplex *d_DCT_coef_x , cufftComplex *d_DCT_coef_y, 
                             cufftComplex *d_IDCT_coef_x, cufftComplex *d_IDCT_coef_y){
  	cufftResult ret = cufftDestroy(*plan);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Plan setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Plan is an invalid handle.\n";
    }
    delete plan;

    cutilSafeCall(cudaFree(d_p_density));
    for(int i=0; i<3; i++)    
        cutilSafeCall(cudaFree(d_p_mode[i]));
    cutilSafeCall(cudaFree(d_t));
    //cutilSafeCall(cudaFree(d_t2));
    cutilSafeCall(cudaFree(d_f_mode));
    cutilSafeCall(cudaFree(d_f_density));
    cutilSafeCall(cudaFree(d_C1));
    cutilSafeCall(cudaFree(d_C2));
    cutilSafeCall(cudaFree(d_DCT_coef_x));
    cutilSafeCall(cudaFree(d_DCT_coef_y)); 
    cutilSafeCall(cudaFree(d_IDCT_coef_x));
    cutilSafeCall(cudaFree(d_IDCT_coef_y)); 
};

void cudagpuBlockMemAlloc( cufftHandle* &plan, 
                           cufftComplex **d_p_density,   cufftComplex *d_p_mode[3],  cufftComplex **d_t, 
                           cufftComplex **d_f_density,   cufftComplex **d_f_mode,
                           cufftComplex **d_C1,          cufftComplex **d_C2,
                           cufftComplex **d_DCT_coef_x,  cufftComplex **d_DCT_coef_y, 
                           cufftComplex **d_IDCT_coef_x, cufftComplex **d_IDCT_coef_y, 
                           int nx, int ny){
	plan = new cufftHandle();
  
    cutilSafeCall(cudaMalloc((void**)d_p_density , sizeof(cufftComplex)*nx*ny));   
    for(int i=0; i<3; i++)    
        cutilSafeCall(cudaMalloc((void**)&(d_p_mode[i])   , sizeof(cufftComplex)*nx*ny));
    cutilSafeCall(cudaMalloc((void**)d_t         , sizeof(cufftComplex)*nx*ny)); 
    //cutilSafeCall(cudaMalloc((void**)d_t2        , sizeof(cufftComplex)*nx*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_f_mode      , sizeof(cufftComplex)*nx*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_f_density   , sizeof(cufftComplex)*nx*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_C1        , sizeof(cufftComplex)*nx*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_C2        , sizeof(cufftComplex)*nx*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_DCT_coef_x  , sizeof(cufftComplex)*nx));
    cutilSafeCall(cudaMalloc((void**)d_DCT_coef_y  , sizeof(cufftComplex)*ny)); 
    cutilSafeCall(cudaMalloc((void**)d_IDCT_coef_x , sizeof(cufftComplex)*nx));
    cutilSafeCall(cudaMalloc((void**)d_IDCT_coef_y , sizeof(cufftComplex)*ny)); 
    cuSetZero<<<BLOCK_NUM, THREAD_NUM>>>(*d_p_density, nx*ny);
    for(int i=0; i<3; i++)    
        cuSetZero<<<BLOCK_NUM, THREAD_NUM>>>(d_p_mode[i], nx*ny);
    cuSetZero<<<BLOCK_NUM, THREAD_NUM>>>(*d_f_density, nx*ny);
    cuSetZero<<<BLOCK_NUM, THREAD_NUM>>>(*d_f_mode, nx*ny);
};

void cuda2dDCTSetCoef ( cufftComplex *d_C1, cufftComplex *d_C2,
                        cufftComplex *d_DCT_coef_x, cufftComplex *d_DCT_coef_y, 
                        cufftComplex *d_IDCT_coef_x, cufftComplex *d_IDCT_coef_y, 
                        int nx, int ny)
{    
    cuC12SetCoef     <<<BLOCK_NUM, THREAD_NUM>>>  (d_C1, d_C2, nx, ny);
    cuDCTSetCoef     <<<BLOCK_NUM, THREAD_NUM>>>  (d_DCT_coef_x, nx);
    cuDCTSetCoef     <<<BLOCK_NUM, THREAD_NUM>>>  (d_DCT_coef_y, ny);
    cuIDCTSetCoef    <<<BLOCK_NUM, THREAD_NUM>>>  (d_IDCT_coef_x, nx);
    cuIDCTSetCoef    <<<BLOCK_NUM, THREAD_NUM>>>  (d_IDCT_coef_y, ny);
};

void cuda2dDCT (cufftHandle* &plan, 
                cufftComplex *d_data_DCT, cufftComplex *d_data, cufftComplex *d_data_t, 
                cufftComplex *d_coef_x, cufftComplex *d_coef_y,  int nx, int ny)
{
    /*
    unsigned int free_mem, total_mem, used_mem;  
    cuMemGetInfo( &free_mem, &total_mem );
    used_mem = total_mem-free_mem;
    printf("before Malloc:total mem: %0.3f MB, free: %0.3f MB, used : %0.3f MB\n",
                ((float)total_mem)/1024.0/1024.0, ((float)free_mem )/1024.0/1024.0, 
                ((float)used_mem )/1024.0/1024.0 ); 
    */
    //first DCT
    cuDCTSwap       <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_data, nx, ny);
    cufftResult ret;
    ret = cufftPlan1d(plan, nx, CUFFT_C2C, ny);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Plan setup failed.\n";
    else if(CUFFT_INVALID_SIZE==ret)
        cout<<"Error: Nx paramter is not a supported size.\n";
    else if(CUFFT_INVALID_TYPE==ret)
        cout<<"Error: The type parameter is not supported..\n";
    else if(CUFFT_ALLOC_FAILED==ret)
        cout<<"Error: Allocation of GPU resources for the plan failed.\n";
    }
    ret = cufftExecC2C((*plan), d_data_t, d_data_DCT, CUFFT_FORWARD);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Exec Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    else if(CUFFT_INVALID_VALUE==ret)
        cout<<"Error: Invalid data value\n";
    else if(CUFFT_EXEC_FAILED==ret)
        cout<<"Error: Exec failed.\n";
    }

    ret = cufftDestroy(*plan);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error:Plan Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    }

    cuDCTCoefMul    <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_DCT, d_coef_x, nx, ny);

    //second DCT
    cuTrans         <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_data_DCT, nx, ny);
    cuDCTSwap       <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_DCT, d_data_t, ny, nx);

    ret = cufftPlan1d(plan, ny, CUFFT_C2C, nx);
    if(CUFFT_SUCCESS!=ret){
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Plan setup failed.\n";
    else if(CUFFT_INVALID_SIZE==ret)
        cout<<"Error: Nx paramter is not a supported size.\n";
    else if(CUFFT_INVALID_TYPE==ret)
        cout<<"Error: The type parameter is not supported..\n";
    else if(CUFFT_ALLOC_FAILED==ret)
        cout<<"Error: Allocation of GPU resources for the plan failed.\n";
    }
    
    ret = cufftExecC2C((*plan), d_data_DCT, d_data_t, CUFFT_FORWARD); 
    if(CUFFT_SUCCESS!=ret){
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Exec Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    else if(CUFFT_INVALID_VALUE==ret)
        cout<<"Error: Invalid data value\n";
    else if(CUFFT_EXEC_FAILED==ret)
        cout<<"Error: Exec failed.\n";
    }
    ret = cufftDestroy(*plan);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error:Plan Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    }

    cuDCTCoefMul    <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_coef_y, ny, nx);
    cuTrans         <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_DCT, d_data_t, ny, nx);
};

void cuda2dIDCT(cufftHandle* &plan, 
                cufftComplex *d_data_IDCT, cufftComplex *d_data, cufftComplex *d_data_t,  
                cufftComplex *d_coef_x, cufftComplex *d_coef_y, int nx, int ny){
    
    //first IDCT
    cuTrans          <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_data, nx, ny);
    cuIDCTCoefMul    <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_coef_y, ny, nx); 

    cufftResult ret = cufftPlan1d(plan, ny, CUFFT_C2C, nx);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Plan setup failed.\n";
    else if(CUFFT_INVALID_SIZE==ret)
        cout<<"Error: Nx paramter is not a supported size.\n";
    else if(CUFFT_INVALID_TYPE==ret)
        cout<<"Error: The type parameter is not supported..\n";
    else if(CUFFT_ALLOC_FAILED==ret)
        cout<<"Error: Allocation of GPU resources for the plan failed.\n";
    }

    ret = cufftExecC2C((*plan), d_data_t, d_data_IDCT, CUFFT_INVERSE);
    if(CUFFT_SUCCESS!=ret){
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Exec Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    else if(CUFFT_INVALID_VALUE==ret)
        cout<<"Error: Invalid data value\n";
    else if(CUFFT_EXEC_FAILED==ret)
        cout<<"Error: Exec failed.\n";
    }
    ret = cufftDestroy(*plan);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error:Plan Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    }

    cuIDCTSwap		<<<BLOCK_NUM, THREAD_NUM>>>  (d_data_t, d_data_IDCT, ny, nx);
    //second IDCT
    cuTrans			<<<BLOCK_NUM, THREAD_NUM>>>  (d_data_IDCT, d_data_t, ny, nx);
    cuIDCTCoefMul	<<<BLOCK_NUM, THREAD_NUM>>>  (d_data_IDCT, d_coef_x, nx, ny);   

    ret = cufftPlan1d(plan, nx, CUFFT_C2C, ny);
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Plan setup failed.\n";
    else if(CUFFT_INVALID_SIZE==ret)
        cout<<"Error: Nx paramter is not a supported size.\n";
    else if(CUFFT_INVALID_TYPE==ret)
        cout<<"Error: The type parameter is not supported..\n";
    else if(CUFFT_ALLOC_FAILED==ret)
        cout<<"Error: Allocation of GPU resources for the plan failed.\n";

    ret = cufftExecC2C((*plan), d_data_IDCT, d_data_t, CUFFT_INVERSE);
    if(CUFFT_SUCCESS!=ret){
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error: Exec Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    else if(CUFFT_INVALID_VALUE==ret)
        cout<<"Error: Invalid data value\n";
    else if(CUFFT_EXEC_FAILED==ret)
        cout<<"Error: Exec failed.\n";
    }
        
    cuIDCTSwap       <<<BLOCK_NUM, THREAD_NUM>>>  (d_data_IDCT, d_data_t, nx, ny);
    ret = cufftDestroy(*plan);
    if(CUFFT_SUCCESS!=ret){    
    if(CUFFT_SETUP_FAILED==ret)
        cout<<"Error:Plan Setup failed.\n";
    else if(CUFFT_INVALID_PLAN==ret)
        cout<<"Error: Invalid plan setting.\n";
    }
};

void cudaUpdatePressure(cufftComplex* d_f_mode, cufftComplex* d_p_mode1, cufftComplex* d_p_mode2, cufftComplex* d_p_mode, 
                        cufftComplex* d_C1, cufftComplex* d_C2, 
                        int width, int height){
    cudaThreadSynchronize();
    cuUpdatePressure	<<<BLOCK_NUM, THREAD_NUM>>>	(d_p_mode1, d_p_mode2, d_p_mode, d_C1, d_C2, d_f_mode, width, height);
    cudaError_t err;
    err=cudaThreadSynchronize();
    //cudaError_t err = cudaPeakLastError();    
    if( cudaSuccess != err) 
        printf( "Cuda error: %s.\n",  cudaGetErrorString( err) );
    //else
    //    printf("Update pressure succeed.\n");
    return;
}


void cudaInterfaceHandling(cufftComplex* d_p, cufftComplex* d_p_neigbor, cufftComplex* d_f, const int start, const int end, const int offset, 
                           const int width, const int height, const int n_width, const int n_height, SiblingType direction){
    switch(direction) { 
        case RIGHT: 
            cuPassForceToRight	<<<BLOCK_NUM, THREAD_NUM>>>	(d_p, d_p_neigbor, d_f, start, end, offset, width, height, n_width, n_height);
            break; 
        case LEFT: 
            cuPassForceToLeft	<<<BLOCK_NUM, THREAD_NUM>>>	(d_p, d_p_neigbor, d_f, start, end, offset, width, height, n_width, n_height);
            break; 
        case UP: 
            cuPassForceToUp		<<<BLOCK_NUM, THREAD_NUM>>>	(d_p, d_p_neigbor, d_f, start, end, offset, width, height, n_width, n_height);
            break; 
        case DOWN: 
            cuPassForceToDown	<<<BLOCK_NUM, THREAD_NUM>>>	(d_p, d_p_neigbor, d_f, start, end, offset, width, height, n_width, n_height);
            break; 
    }
    return;
}

void cudaUpdateSoundSource(cufftComplex* d_f, const int x, const int y, const int width, const int height, const float src){
    
    //column-wise and row-wise indexing transfer so that interchange x and y 
    cuUpdateSoundSource	<<<BLOCK_NUM, THREAD_NUM>>>	(d_f, y, x, width, height, src);
    return;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
};

static __global__ void cuUpdateSoundSource(cufftComplex* d_f, const int x, const int y, const int width, const int height, const float src){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = threadID; j < width*height; j += numThreads){
        int m=j%width;
        int n=j/width;
        if(m==x && n==y){
            d_f[m+n*width].x=src*FNORM_FACTOR; 
            d_f[m+n*width].y=0.0f;
        }
    }
    __syncthreads();
};

static __global__ void cuPassForceToUp(cufftComplex* d_p, cufftComplex* d_p_neigbor, cufftComplex* d_f, 
                                       const int start, const int end, const int offset, const int width, const int height, const int n_width, const int n_height)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = threadID+start; j < end; j += numThreads){
		float Si = - 2.0 * d_p[2 * width + j].x
                   + 27.0 * d_p[1 * width + j].x
                   - 270.0 * d_p[j].x
				   + 270.0 * d_p_neigbor[(n_height - 1) * n_width + (j + offset)].x
				   - 27.0 * d_p_neigbor[(n_height - 2) * n_width + (j + offset)].x
				   + 2.0 * d_p_neigbor[(n_height - 3) * n_width + (j + offset)].x;
		d_f[j].x = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
	}
    __syncthreads();
};


static __global__ void cuPassForceToLeft(cufftComplex* d_p, cufftComplex* d_p_neigbor, cufftComplex* d_f, 
                                         const int start, const int end, const int offset, const int width, const int height, const int n_width, const int n_height)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = threadID+start; j < end; j += numThreads){
		float Si = - 2.0 * d_p[j * width + 2].x
                   + 27.0 * d_p[j * width + 1].x
                   - 270.0 * d_p[j * width].x
				   + 270.0 * d_p_neigbor[(j+offset) * n_width + (n_width - 1 )].x
				   - 27.0 * d_p_neigbor[(j+offset) * n_width  + (n_width - 2 )].x
				   + 2.0 * d_p_neigbor[(j+offset) * n_width  + (n_width - 3 )].x;
		d_f[j * width].x = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
	}
    __syncthreads();
};

static __global__ void cuPassForceToRight(cufftComplex* d_p, cufftComplex* d_p_neigbor, cufftComplex* d_f, 
                                          const int start, const int end, const int offset, const int width, const int height, const int n_width, const int n_height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = threadID+start; j < end; j += numThreads){
		float Si = - 2.0 * d_p[ j * width + (width - 3)].x
                   + 27.0 * d_p[ j * width + (width - 2)].x
                   - 270.0 * d_p[ j * width + (width - 1)].x
				   + 270.0 * d_p_neigbor[(j+offset) * n_width].x
				   - 27.0 * d_p_neigbor[(j+offset) * n_width + 1].x
				   + 2.0 * d_p_neigbor[(j+offset) * n_width + 2].x;
		d_f[j * width + (width-1)].x = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
	}
    __syncthreads();
};

static __global__ void cuPassForceToDown(cufftComplex* d_p, cufftComplex* d_p_neigbor, cufftComplex* d_f, 
                                         const int start, const int end, const int offset, const int width, const int height, const int n_width, const int n_height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = threadID+start; j < end; j += numThreads){
		float Si = - 2.0 * d_p[ (height - 3) * width + j].x
                   + 27.0 * d_p[ (height - 2) * width + j].x
                   - 270.0 * d_p[ (height - 1) * width + j].x
				   + 270.0 * d_p_neigbor[ (j+offset) ].x
				   - 27.0 * d_p_neigbor[1 * n_width + (j+offset)].x
				   + 2.0 * d_p_neigbor[2 * n_width + (j+offset)].x;
		d_f[(height-1) * width + j].x = Si * SOUND_SPEED * SOUND_SPEED / (180.0 * H * H);
	}
    __syncthreads();
};

static __global__ void cuUpdatePressure(cufftComplex* d_p_mode1, cufftComplex* d_p_mode2, cufftComplex* d_p_mode3, 
                                        cufftComplex* d_C1, cufftComplex* d_C2, cufftComplex* d_f_mode, const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < width*height; i += numThreads){
        d_p_mode3[i].x = (d_C1[i].x * d_p_mode1[i].x) + (d_f_mode[i].x * d_C2[i].x ) - d_p_mode2[i].x;
        d_p_mode3[i].y = 0.0f;
    }
    __syncthreads();
};

static __global__ void cuDCTSetCoef(cufftComplex* coef, const int width){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < width; i += numThreads){
        coef[i].x =  __cosf(CUDA_PI*i/(2*width));
        coef[i].y = -__sinf(CUDA_PI*i/(2*width));  
    }
    __syncthreads();
};

static __global__ void cuC12SetCoef(cufftComplex* d_C1, cufftComplex* d_C2, const int colSize, const int rowSize){
    //initial update constants
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  	float lx = colSize * H;
	float ly = rowSize * H;
    int x,y;
    for (int i = threadID; i < rowSize*colSize; i += numThreads){
        x=i%colSize;
        y=i/colSize;
        if(x==0&&y==0) {
            d_C1[x+y*colSize].x=1.0; 
            d_C2[x+y*colSize].x=0.0;
            d_C1[x+y*colSize].y=0.0; 
            d_C2[x+y*colSize].y=0.0;
        }
        else{
            float omega = SOUND_SPEED * CUDA_PI * sqrtf((float)x*(float)x/lx/lx + (float)y*(float)y/ly/ly);
    		float cosomega = __cosf(omega*DELTA_T);
            d_C1[x+y*colSize].x=2.0*cosomega;
            d_C1[x+y*colSize].y=0.0;
            d_C2[x+y*colSize].x=2.0 * (1.0-cosomega) / omega /omega; 
            d_C2[x+y*colSize].y=0.0;     
        }
       
    }
    __syncthreads();
};


static __global__ void cuIDCTSetCoef(cufftComplex* coef, const int width){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < width; i += numThreads){
            coef[i].x = __cosf(CUDA_PI*i/(2*width));
            coef[i].y = __sinf(CUDA_PI*i/(2*width));  
    }
    __syncthreads();
};

static __global__ void cuDCTSwap(cufftComplex* data_y, cufftComplex* data,  const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int m, n;
    for (int i = threadID; i < width*height; i += numThreads){
        m = i % width;
        n = i / width;
        if(m%2 == 0)
            data_y[ n*width + m/2 ] = data[ n*width + m ];
        else
            data_y[ n*width + width-1-(m-1)/2] = data[ n*width + m ];
    }        
    __syncthreads();

};

static __global__ void cuIDCTSwap(cufftComplex* data, cufftComplex* data_y,  const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int m, n;
    for (int i = threadID; i < width*height; i += numThreads){
        m = i % width;
        n = i / width;
        if(m%2 == 0){
            data[ n*width + m ].x = (data_y[ n*width + m/2].x)/(width);
            data[ n*width + m ].y = 0;
        }
        else{
            data[ n*width + m ].x = (data_y[ n*width + width-1-(m-1)/2].x) / (width) ;
            data[ n*width + m ].y = 0;
        }
    }        
    __syncthreads();
};

static __global__ void cuDCTCoefMul(cufftComplex* data, cufftComplex* coef, const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int m;
    for (int i = threadID; i < width*height; i += numThreads){
        m = i%width;
        if( m == 0)
            data[i].x = ComplexMul(data[i], coef[m]).x; // real part
        else
            data[i].x = ComplexMul(data[i], coef[m]).x * 2;
        data[i].y = 0;
    }        
    __syncthreads();
};

static __global__ void cuIDCTCoefMul(cufftComplex* data, cufftComplex* coef, const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int m;
    for (int i = threadID; i < width*height; i += numThreads){
        m = i % width;
        data[i] = ComplexMul(data[i], coef[m]);
    }
    __syncthreads();
};

static __global__ void cuTrans (cufftComplex* data_trans, cufftComplex* data, const int width, const int height){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int m, n;
    for (int i = threadID; i < width*height; i += numThreads){
        m = i%width;
        n = i/width;
        data_trans[ m*height + n] = data[ n*width + m]; // real part
    }        
    __syncthreads();
};

static __global__ void cuSetZero(cufftComplex* d_data, const int length){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < length; i += numThreads){
        d_data[i].x=0.0;
        d_data[i].y=0.0;
    }
    __syncthreads();
};

#endif
