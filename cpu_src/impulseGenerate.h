#include <iostream>
#include <cmath>

using namespace std;
#define PI 3.14159265358979323846

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
