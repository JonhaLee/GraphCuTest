#include "filePath.h"

filePath* filePath::instance = nullptr;

filePath* filePath::getInstance(){
	if (instance == nullptr)
		instance = new filePath();

	return instance;
}

const std::string filePath::getColorPath(int number){
	std::string colorfileName = "KinectScreenshot_RGB";
	
	std::string colorPath = filepath +dataName + "\\Color\\" + colorfileName + std::to_string(number) + ".bmp";
	
	return colorPath;
}

const std::string filePath::getBodyIndexPath(int number){
	std::string colorfileName = "FileHRbodyIndex_";

	std::string colorPath = filepath + dataName + "\\HR_BodyIndex\\" + colorfileName + std::to_string(number) + ".bin";

	return colorPath;
}

const std::string filePath::getMappPath(int number){
	std::string colorfileName = "FileMapp_";

	std::string colorPath = filepath + dataName + "\\Mapp\\" + colorfileName + std::to_string(number) + ".bin";

	return colorPath;
}

const std::string filePath::getDepthPath(int number){
	std::string colorfileName = "Filedepth_";

	std::string colorPath = filepath + dataName + "\\Depth\\" + colorfileName + std::to_string(number) + ".bin";

	return colorPath;
}
