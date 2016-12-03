#include "filePath.h"

filePath* filePath::instance = nullptr;

filePath* filePath::getInstance(){
	if (instance == nullptr)
		instance = new filePath();

	return instance;
}

const std::string filePath::getColorPath(int number){
	std::string fileName = "KinectScreenshot_RGB";
	
	std::string path = filepath + dataName + "\\Color\\" + fileName + std::to_string(number) + ".bmp";
	
	return path;
}

const std::string filePath::getBodyIndexPath(int number){
	std::string fileName = "FileHRbodyIndex_";

	std::string path = filepath + dataName + "\\HR_BodyIndex\\" + fileName + std::to_string(number) + ".bin";

	return path;
}

const std::string filePath::getMappPath(int number){
	std::string fileName = "FileMapp_";

	std::string path = filepath + dataName + "\\Mapp\\" + fileName + std::to_string(number) + ".bin";

	return path;
}

const std::string filePath::getDepthPath(int number){
	std::string fileName = "Filedepth_";

	std::string path = filepath + dataName + "\\Depth\\" + fileName + std::to_string(number) + ".bin";

	return path;
}

const char* filePath::getSkeletonPath(){
	std::string fileName = "Fileskeleton";

	std::string path = filepath + dataName + "\\Body\\" + fileName + ".csv";

	return _strdup(path.c_str());	
}

const std::string filePath::getResultPath(){
	std::string path = filepath + dataName + "\\Result\\";

	return path;
}