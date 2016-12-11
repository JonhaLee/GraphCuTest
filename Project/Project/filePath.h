#pragma once

#include <string>

//////////////////////////////////////////////////////////////////////////
//여기서 path 설정
static std::string filepath = "C:\\Users\\Jonha\\Desktop\\";
static std::string dataName = "Data9";
//////////////////////////////////////////////////////////////////////////

class filePath{
private:
	filePath(){}
	static filePath* instance_;

public:
	static filePath* getInstance();	
	static const std::string getColorPath(int number);
	static const std::string getBodyIndexPath(int number);
	static const std::string getMappPath(int number);
	static const std::string getDepthPath(int number);
	static const char* getSkeletonPath();
	static const std::string getResultPath();
};
