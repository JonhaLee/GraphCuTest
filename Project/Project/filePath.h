#pragma once

#include <string>

//////////////////////////////////////////////////////////////////////////
//���⼭ path ����
static std::string filepath = "C:\\Users\\Jonha\\Desktop\\";
static std::string dataName = "Data8";
//////////////////////////////////////////////////////////////////////////

class filePath{
private:
	filePath(){}
	static filePath* instance;

public:
	static filePath* getInstance();	
	static const std::string getColorPath(int number);
	static const std::string getBodyIndexPath(int number);
	static const std::string getMappPath(int number);
	static const std::string getDepthPath(int number);
};
