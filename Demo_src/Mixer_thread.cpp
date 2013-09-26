extern "C"{
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>

#include <pulse/simple.h>
#include <pulse/error.h>
#include <pulse/gccmacro.h>
}

#include <cv.h>
#include <iostream>
#include <fstream>
#include <ctime>

#include "TruncateIR.h"
#include "filelock.hpp"

using namespace std;
using namespace cv;

#define BUFSIZE 32768

pa_simple *s = NULL;
pa_simple *s1 = NULL;
short buf[BUFSIZE] = {0};
short buf2[BUFSIZE*2] = {0};
short impulse[BUFSIZE*2] = {0};
short reserve[BUFSIZE] = {0};

bool getNewData = false;

Mat AudioIn_f, Response, AudioIn_fft, Response_fft, Output_fft, Output;
Mat  Output_s16(1, BUFSIZE*2, CV_16SC1);

void* RecCallback(void* arg)
{
	while(1){
		while(!getNewData){}

		clock_t start = clock();
		//cpy
		memcpy(buf2, buf, BUFSIZE*sizeof(short));
		getNewData = false;

		int error;
		ssize_t r;
		
		Mat AudioIn(1, BUFSIZE*2, CV_16SC1, buf2);
		Mat AudioIn_f;
		AudioIn.convertTo(AudioIn_f, CV_32FC1);
//		impulse[0] = 3500;
//		impulse[2000] = 3500;
		Response = Mat(1, BUFSIZE*2, CV_16SC1, impulse);
		Mat Response_f;
		Response.convertTo(Response_f, CV_32FC1);

		//Mat AudioIn_fft;
		dft(AudioIn_f, AudioIn_fft, DFT_ROWS);

		//Mat Response_fft;
		dft(Response_f, Response_fft, DFT_ROWS);

		//Mat Output_fft;

		mulSpectrums(AudioIn_fft, Response_fft, Output_fft, DFT_ROWS);

		//Mat Output;
		dft(Output_fft, Output, DFT_ROWS | DFT_INVERSE | DFT_SCALE);
		Output = Output / 10000;
		Output.convertTo(Output_s16, CV_16SC1);

		//cout << Output_s16.rows << " " << Output_s16.cols << endl;
		for(int i = 0; i < BUFSIZE; ++i){
			Output_s16.at<short>(0,i) = Output_s16.at<short>(0,i) + reserve[i];
			reserve[i] = Output_s16.at<short>(0,i+BUFSIZE);
		}
		if (r = pa_simple_write(s, (char*)Output_s16.data, BUFSIZE*sizeof(short), &error) < 0) {
			fprintf(stderr, __FILE__": pa_simple_write() failed: %s\n", pa_strerror(error));
		}
		clock_t end = clock();
		printf("write time : %d\n", int((end-start)*1000/CLOCKS_PER_SEC));
	}
}


void* ReadFunction(void* arg)
{
	while(1){
		int error;
		ssize_t r;
        if (r = pa_simple_read(s1, buf, BUFSIZE*2, &error) < 0) {
            fprintf(stderr, __FILE__": pa_simple_read() failed: %s\n", pa_strerror(error));
        }
        printf("%d\n", r);

        pa_usec_t latency;

        if ((latency = pa_simple_get_latency(s, &error)) == (pa_usec_t) -1) {
            fprintf(stderr, __FILE__": pa_simple_get_latency() failed: %s\n", pa_strerror(error));
        }
        //fprintf(stderr, "%0.0f usec    \r", (float)latency);
		printf("latency now %0.0f usec \n", (float)latency);

        getNewData = true;
        /* ... and play it */

        while(getNewData){}
	}
}

short** IR;
int sub_rows;
int sub_cols;
int recv_x = 0;
int recv_y = 0;
int play = 0;

void* ReadQTThread(void* arg)
{
	while(1){
		while(!checkFilelock("lockfile")){}
		getFilelock("lockfile");
		ifstream fin("positionfile");
		int x,y,p;
		fin >> y >> x >> p;
		fin.close();
		releaseFilelock("lockfile");
		if(x != recv_x || y != recv_y || p != play){
			if(x/16 > sub_rows || y/16 > sub_cols) continue;
			cout << "QT thread"<< " " << x << " " << y << " " << p <<endl;
			recv_x = x;
			recv_y = y;
			play = p;
			short* imbuf = new short[BUFSIZE];
			if(play == 0){
				cout << "stop" << endl;
				memset((char*)imbuf, 0, BUFSIZE*sizeof(short));
			}else if(play == 1){
				cout << "play " <<endl;
				memcpy((char*)imbuf, (char*)(IR[(recv_x/16)*sub_cols + (recv_y/16)]), BUFSIZE*sizeof(short));
			}else if(play == 2){
				cout << "origin" << endl;
				memset((char*)imbuf, 0, BUFSIZE*sizeof(short));
				imbuf[0] = 7000;
			}
			TruncateIR(imbuf, BUFSIZE);
			memcpy((char*)impulse, (char*)imbuf, BUFSIZE*sizeof(short));
			delete[] imbuf;
		}
		sleep(1);
	}
}

int latency = 800000;

int main(int argc, char*argv[]) {

    /* The Sample format to use */
    static pa_sample_spec ss;
    ss.format = PA_SAMPLE_S16LE;
    ss.rate = 22050;
    ss.channels = 1;
	
	pa_buffer_attr bufattr;
	bufattr.fragsize = (uint32_t)-1;
	bufattr.maxlength = pa_usec_to_bytes(latency,&ss);
	bufattr.minreq = pa_usec_to_bytes(0,&ss);
	bufattr.prebuf = (uint32_t)-1;
	bufattr.tlength = pa_usec_to_bytes(latency,&ss);

    int ret = 1;
    int error;

    /* Create a new playback stream */
	/* Create the recording stream */
	
    if (!(s1 = pa_simple_new(NULL, argv[0], PA_STREAM_RECORD, NULL, "record", &ss, NULL, NULL, &error))) {
        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
        goto finish;
    }
    if (!(s = pa_simple_new(NULL, argv[0], PA_STREAM_PLAYBACK, NULL, "playback", &ss, NULL, &bufattr, &error))) {
        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
        goto finish;
    }
	
	//read impulse response
	if(argc > 1){
		cout << "reading IR..." <<endl;
		ifstream IRfin(argv[1], ios::binary);
		if(!IRfin) return -1;
		int duration;
		IRfin.read((char*)&duration, sizeof(int));
		IRfin.read((char*)&sub_rows, sizeof(int));
		IRfin.read((char*)&sub_cols, sizeof(int));
		IR = new short*[sub_rows*sub_cols];
		for(int i = 0; i < sub_rows*sub_cols; ++i){
			IR[i] = new short[duration];
			IRfin.read((char*)(IR[i]), duration* sizeof(short));
		}
		IRfin.close();
		cout << "reading IR done." <<endl;
	}else{
		impulse[0] = 7000;
	}

	pthread_t q_thread;
	if(pthread_create(&q_thread, NULL, ReadQTThread, NULL) != 0)
		fprintf(stderr, "Error creating the thread");

	pthread_t p_thread;
 	if(pthread_create(&p_thread, NULL, RecCallback, NULL) != 0)
 		fprintf(stderr, "Error creating the thread");

	pthread_t p_thread2;
 	if(pthread_create(&p_thread2, NULL, ReadFunction, NULL) != 0)
 		fprintf(stderr, "Error creating the thread");


	pthread_join(q_thread, NULL);
	pthread_join(p_thread, NULL);
	pthread_join(p_thread2, NULL);



/*
    for (;;) {
        ssize_t r;

#if 0
        pa_usec_t latency;

        if ((latency = pa_simple_get_latency(s, &error)) == (pa_usec_t) -1) {
            fprintf(stderr, __FILE__": pa_simple_get_latency() failed: %s\n", pa_strerror(error));
            goto finish;
        }

        fprintf(stderr, "%0.0f usec    \r", (float)latency);
#endif

        // Read some data ... 
		if (r = pa_simple_read(s1, buf, BUFSIZE*2, &error) < 0) {
            fprintf(stderr, __FILE__": pa_simple_read() failed: %s\n", pa_strerror(error));
            goto finish;
        }
		printf("%d\n", r);

		getNewData = true;
        // ... and play it 

		while(getNewData){}
        
    }*/

	

    /* Make sure that every single sample was played */
    if (pa_simple_drain(s, &error) < 0) {
        fprintf(stderr, __FILE__": pa_simple_drain() failed: %s\n", pa_strerror(error));
        goto finish;
    }

    ret = 0;

finish:

    if (s)
        pa_simple_free(s);
	if (s1)
		pa_simple_free(s1);

    return ret;
}

