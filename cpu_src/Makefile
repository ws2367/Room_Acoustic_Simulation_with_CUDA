BINDIR := ../bin/release

OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann 

OPENCV_FLAGS = -I/usr/local/include/opencv -I/usr/local/include 

all: gpu_config.h gpu_block.h partitionMap.h main.cpp
	g++ $(OPENCV_LIBS) $(OPENCV_FLAGS)  main.cpp -o $(BINDIR)/Simulation_cpu
clean:
	rm -f $(BINDIR)/Simulation_cpu
