#ifndef TEST_TOOL_H
#define TEST_TOOL_H
void getForce(cufftComplex*& d_p_density, cufftComplex* h_p, const int colSize, const int rowSize, char name){
    if(h_p==NULL) cout<<"Error: Memory allocation failed in reading pressure field on GPU memory.\n";
    cutilSafeCall(cudaMemcpy(h_p, d_p_density, sizeof(cufftComplex)*colSize*rowSize, cudaMemcpyDeviceToHost));
    //cout<<"get Pressure field over.\n";

    double max=0, min=0;
    for(int i=0;i<colSize*rowSize; i++){
        if(max<h_p[i].x) max=h_p[i].x;
        if(min>h_p[i].x) min=h_p[i].x;
    }
    printf("%c max:%e, min:%e.\n",name,max,min);       
    return;
}

void seeCoef(cufftComplex* d_data, const int length){
    cufftComplex* h_data = (cufftComplex*) malloc(sizeof(cufftComplex)*length);
    cutilSafeCall(cudaMemcpy(h_data, d_data, sizeof(cufftComplex)*length, cudaMemcpyDeviceToHost));
    double max=0, min=0;
    for(int i=0; i<length; i++){
        printf("%e ",h_data[i].x);
        if(max<h_data[i].x) max=h_data[i].x;
        if(min>h_data[i].x) min=h_data[i].x;
        if(i%5==0&&i!=0) cout<<endl;
    }
    cout<<endl;
    printf("max:%e, min:%e\n)",max,min);
    free(h_data);
    return;
}

#endif
