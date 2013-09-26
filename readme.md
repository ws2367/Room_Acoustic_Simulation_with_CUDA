Room Acoustic Simulation with CUDA
===================================

Wei-Chih Hung, Wen-Hsiang Shaw and Yen-Cheng Chou 

##Introduction

We implemented a 2D room acoustic simulation system. The system is able to simulate the sound for a listener at a designated position in the given room map with the multiple sound sources. The 2D room map and the positions of the sound sources need to be designated in the process of pre-computed simulation. With the acceleration of CUDA, the simulation process can be shorten a lot. The system allows changing the position of the listener and the sound sources input in real time. The propagation is simulated in frequency domain by exploiting the Acoustic Wave Equation via numerical approach.

Content
--------
* project : The whole project source file and executables
* project/Docs : the presentation sides and detailed documentation
* project/gpu_src : Simulation with CUDA source file
* project/cpu_src : Simulation in CPU source file
* project/RoomModel : example simulation room models
* project/DemoGui_src : real-time demo tool's gui source file
* project/Demo_src : real-time demo tool's background source file
* project/Demo : pre-build executables and simulation results(ubuntu 32bit)
* project/bin : compile destination folders
* project/prebuild_bin : prebuild binarys in linux 32-bits


How To Compile
--------------
To use the built binaries, copy the project/prebuild_bin to project/bin. The dynamic libraries are required to be linked.

Request library :
* CUDA SDK : we use CUDA SDK 3.2 and CUFFT
* OPENCV : 2.2+ library
* QT : QT4 for demo tools GUI
* pulse-audio : this library is usually pre-installed in current linu
