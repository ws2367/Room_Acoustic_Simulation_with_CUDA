#ifndef TRUNCATEIR_H
#define TRUNCATEIR_H
#include <iostream>
#include <cmath>

void TruncateIR(short* IR, int length)
{
	int lmin = IR[0];
	int lmax = 0;
	int lmaxIdx = 0;
	bool findmax = true;
	float* scale_factor = new float[length];
	int r_max=0;
	for(int i = 0; i < length; ++i){
		scale_factor[i] = exp(-(float)i*5/length);
		if(IR[i] > r_max) r_max = IR[i];
	}

	for(int i = 1; i < length-1; ++i){
		if(findmax){
			if(IR[i+1] < IR[i]){
				lmax = IR[i];
				lmaxIdx = i;
				findmax = false;
			}else{
				IR[i] = 0;
			}
		}else{
			if(IR[i+1] > IR[i]){
				int diff = 2*lmax - IR[i] - lmin;
				if(diff > 0.7 * r_max && lmax > 0)
					IR[lmaxIdx] = lmax;
				else
					IR[lmaxIdx] = 0;
				lmin = IR[i];
				findmax = true;
			}
			IR[i] = 0;
		}
	}

	for(int i = 0; i < length; ++i)
		IR[i] = scale_factor[i] * IR[i];
}

#endif 
