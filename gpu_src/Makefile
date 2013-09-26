################################################################################
#
# Build script for project
#
################################################################################

CUDA_SDK_PATH := /home/lamb/NVIDIA_GPU_Computing_SDK

#cuda compute sdk root dir 
ROOTDIR := $(CUDA_SDK_PATH)/C/src/

#binary DIR
BINDIR := ../bin

# Add source files here
EXECUTABLE	:= Simulation_gpu
# Cuda source files (compiled with cudacc)
CUFILES		:= main.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
# Additional libraries needed by the project
USECUFFT        := 1


################################################################################
# Rules and targets

include $(ROOTDIR)/../common/common.mk

OPENCV_INC_PATH := /usr/local/include/opencv
OPENCV_LIB_PATH := /usr/local/lib
ADD_INC_PATH := /usr/local/include
OPENCV_LIB_FLAG := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

# for OpenCV's include and libs, create path
NVCCFLAGS += -I$(OPENCV_INC_PATH) -I$(ADD_INC_PATH) -arch sm_13
LIB += -L$(OPENCV_LIB_PATH) $(OPENCV_LIB_FLAG)
CXXFLAGS += -I$(OPENCV_INC_PATH) -I$(ADD_INC_PATH)

