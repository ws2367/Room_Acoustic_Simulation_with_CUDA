#ifndef FILE_LOCK_H
#define FILE_LOCK_H

#include <fstream>
#include <iostream>

using namespace std;

bool checkFilelock(const char* lockfile)
{
	ifstream testlock(lockfile);
	if(testlock){
	
#ifdef DEBUG
		printf("Filelock \"%s\" already exist! \n", lockfile);
#endif
		return false;
	}else{
	
#ifdef DEBUG
		printf("File lock \"%s\" check ok \n", lockfile);
#endif
		return true;
	}

}

void getFilelock(const char* lockfile)
{
	ofstream lock(lockfile);
	#ifdef DEBUG
	printf("get file lock %s \n", lockfile);
	#endif
}

void releaseFilelock(const char* lockfile)
{
	remove(lockfile);
	#ifdef DEBUG
	printf("release file lock %s \n", lockfile);
	#endif
}

#endif
