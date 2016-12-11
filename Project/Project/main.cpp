#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "BinaryStream.h"

#include "gc.h"
#include "filePath.h"

//�׽�Ʈ�� ���� �ڵ�
#include <time.h>

using namespace cv;

//�����
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define PIE 3.1419

const static int kTotal_BodyIndex = 6;
const static int kJointFromKinectV2 = 25;

bool isBody[kTotal_BodyIndex];


//File�κ��� Data�� �о���� �Լ���
BYTE* loadBodyIndexFile(int frameNumber);
short* loadMappingFile(int frameNumber);
BYTE* loadDepthImageFile(int frameNumber);
Point2i** loadSkeletonFile(int frameNumber);


//Seed�� �Է��ϴ� �Լ���
void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData);
void addSeedBySkeleton(Mat* mask, BYTE** skeletonData);


//Skeleton ���� ���� �Լ���
BYTE** createSkeletonData(int frameNumber, Point2i** skeletonDatas_origin);
Mat* createSkeletonWeightMap(int frame, BYTE** skeletonDatas);
void CreateSkeletonLines(Point2i* jointPoints, int width, int height, int stroke, BYTE* skeletonMap);


//��ƿ��Ƽ �Լ���
Point2i mappingLowToHigh(Point2i point, short* mappingData);
void expandPixelBy(int x, int y, int width, int height, int stroke, BYTE* skeletonDatas);
void drawLineBy(Point2i start, Point2i end, int width, int height, int stroke, BYTE* skeletonDatas);

const char* getfield(char* line, int num);



int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	//TEST: �ҿ�ð� ����(begin)//////////////
	clock_t begin, end;
	begin = clock();
	//////////////////////////////////////////


	//ó���ϰ��� �ϴ� ������ frame�� �Է�(�ݺ��� ����� ���� ������)
	const int kFrameNumber = 70;


	//BodyIndexData�� ��� ���� local files�� �о�´�.
	/**/printf("%d��° ������, BodyIndexData �ε� ����\n", kFrameNumber);
	BYTE* bodyIndexData = loadBodyIndexFile(kFrameNumber);
	/**/printf("%d��° ������, BodyIndexData �ε� �Ϸ�\n", kFrameNumber);

	/*
	GraphCut�� ��� �� Seed(foreground or background)�� �Է�
	kinect���� �������ִ� BodyIndex ������ŭ(����� 6) GraphCut�� ������ ���� ������ Seed Mask�� ����
	�⺻������ ��� Seed�� PR_Background�� �־��� ������, �ڿ��� PR_Foreground�� Foreground�� �ٽ� �Է�
	*/
	/**/printf("%d��° ������, BodyIndexData Seed�� �ֱ� ����\n", kFrameNumber);
	Mat* GC_Masks = new Mat[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		GC_Masks[i] = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(GC_PR_BGD));
	}	
	addSeedByBodyIndex(GC_Masks, bodyIndexData);	
	//�޸� ����
	delete[] bodyIndexData;
	/**/printf("%d��° ������, BodyIndexData Seed�� �ֱ� �Ϸ�\n", kFrameNumber);


	/*
	���̷��� ���� �����κ�
	���̷��� joint ������ ��� ���� local file(.csv)�κ��� ������ ������
	�ش� joint ������ �������� ����(line)���� �����ϴ� �κ�
	*/	
	/**/printf("%d��° ������, ���̷��� csv ���� �ε� ����\n", kFrameNumber);
	Point2i** skeletonDatas_origin = loadSkeletonFile(kFrameNumber);
	/**/printf("%d��° ������, ���̷��� csv ���� �ε� �Ϸ�\n", kFrameNumber);


	/**/printf("%d��° ������, ���̷��� Data ���� ����\n", kFrameNumber);
	BYTE** skeletonDataMaps = createSkeletonData(kFrameNumber, skeletonDatas_origin);
	/**/printf("%d��° ������, ���̷��� Data ���� �Ϸ�\n", kFrameNumber);

		
	/**/printf("%d��° ������, SkeletonData�� �õ� �ֱ� ����\n", kFrameNumber);
	addSeedBySkeleton(GC_Masks, skeletonDataMaps);
	/**/printf("%d��° ������, SkeletonData�� �õ� �ֱ� �Ϸ�\n", kFrameNumber);


	//���� SkeletonWiethMap�� ����� �κ�
	/**/printf("%d��° ������, SkeletonData�� ����ġ �� ���� ����\n", kFrameNumber);
	Mat* skeletonMaps = createSkeletonWeightMap(kFrameNumber, skeletonDataMaps);
	/**/printf("%d��° ������, SkeletonData�� ����ġ �� ���� �Ϸ�\n", kFrameNumber);
		


	//�Է� ����
	Mat con_img = imread(filePath::getInstance()->getColorPath(kFrameNumber));

	
	/*
	�ʿ��� image ���ϵ��� ����� �κ�
	*/

	//�̹��� ���� �������� ��¥�� �ð��� ����ֱ� ����
	struct tm* datetime;
	time_t t;
	t = time(NULL);
	datetime = localtime(&t);

	std::string date = 
		to_string(datetime->tm_year + 1900) + "-"
		+ to_string(datetime->tm_mon + 1) + "-"
		+ to_string(datetime->tm_mday) + "-"
		+ to_string(datetime->tm_hour) + "-"
		+ to_string(datetime->tm_min) + "-"
		+ to_string(datetime->tm_sec);

	//������ ���� path
	std::string Spath_inputImage			= filePath::getInstance()->getResultPath() + "input" + to_string(kFrameNumber) + "_" + date + ".jpg";
	std::string Spath_depthNskeletonImage	= filePath::getInstance()->getResultPath() + "depth_skeleton" + to_string(kFrameNumber) + "_" + date + ".jpg";
	std::string Spath_HRskeletonImage = filePath::getInstance()->getResultPath() + "HRskeleton" + to_string(kFrameNumber) + "_" + date + ".jpg";

		
	//���� �� + skeleton ó�� �κ�
	BYTE* depthData		= loadDepthImageFile(kFrameNumber);
	BYTE* skeleton_origin_map = new BYTE[424 * 512];

	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (isBody[i]){			
			CreateSkeletonLines(skeletonDatas_origin[i], 512, 424, 3, skeleton_origin_map);
		}
	}
		
	Mat depthNskeleton(424, 512, CV_8UC1, Scalar(0));

	for (int row = 0; row < 424; row++){
		for (int col = 0; col < 512; col++){
			depthNskeleton.at<uchar>(row, col) = depthData[row * 512 + col];
			if (skeleton_origin_map[row * 512 + col] == 1){				
				depthNskeleton.at<uchar>(row, col) = 0;
			}

		}
	}
	
	Mat HRSkeleton(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(255));
	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDataMaps[i][row * IMAGE_WIDTH + col] == 1){
					//printf("%d * 512 + %d\n", row, col);
					HRSkeleton.at<uchar>(row, col) = 0;
				}
			}
		}
	}


	//������ �ϴ� �Լ�
	imwrite(Spath_inputImage.c_str(), con_img);			//�Է� ����
	imwrite(Spath_depthNskeletonImage.c_str(), depthNskeleton);	//���� �� + skeleton
	imwrite(Spath_HRskeletonImage.c_str(), HRSkeleton);
	
	//�޸� ����
	delete[] depthData;
	delete[] skeleton_origin_map;




	//�־��� ������ ���� GraphCut�� ������ �κ�

	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (isBody[i]){
			/**/printf("%d��° ������, %d��° BodyIndex GraphCut ����\n", kFrameNumber, i);
			
			//GrabCut�� ���Ǵ� ������
			//���⼭�� GrabCut�� GraphCutó�� �̿��ϱ� ������ �Ʒ� �������� ���ϰ����� ���� ������� ����.
			my::GraphCut gc;
			Mat back, fore;	//��(�ʱ� ���)
			Rect rect(10, 10, 100, 100);

			//�����δ� GraphCutó�� �۵�
			gc.graphCut(con_img,		//�Է¿���		
				GC_Masks[i],			//���� ����ũ, �ش� ����ũ�� ����� ������� GraphCut ����
				rect,					//������ �����ϴ� ���簢��, ���⼭�� MASK ��带 ��������� rect�� ���� ���� ������� �ʴ´�.
				back, fore,				//��
				1,						//�ݺ�Ƚ��
				GC_INIT_WITH_MASK,		//Mask�� ����ϴ� ��� ����
				skeletonMaps[i]);		//Skeleton Weight Map


			/*
			GraphCut�� ��������� ������ GC_Masks���� ���ϴ� ������� �̾Ƽ�(compare�� ����) �����ϱ� ���� ����
			*/
			Mat fgd_result, pr_fgd_result;

			/*
			���������� �̹��� ���Ϸ� �����ϱ� ���� ������
			*/
			//Mat foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			//Mat pr_foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));

			//��� ���󿡼� GC_FGD�� �ȼ��鸸 fgd_result�� ����
			compare(GC_Masks[i], cv::GC_FGD, fgd_result, cv::CMP_EQ);
			//�� �Ŀ� �Է� ����� ���Ͽ� foreground��� ������ �����Ǵ� �÷� ���� ����
			//��, GC_FGD��� ���� ����� ��ġ�� �÷� ���� ����
			//con_img.copyTo(foreground, fgd_result);

			//���������� GC_PR_FGD�� ���� ����
			compare(GC_Masks[i], GC_PR_FGD, pr_fgd_result, CMP_EQ);
			//con_img.copyTo(pr_foreground, pr_fgd_result);

			//���� ��� ����(final_result)�� fgd, pr_fgd�� �÷� ������ ����
			con_img.copyTo(final_result, fgd_result);
			con_img.copyTo(final_result, pr_fgd_result);

			//final_result ����� skeletonData�� ���� ���ļ� �÷� ���� ���� skeleton line�� ���̰� ǥ���ϴ� �۾�
			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonDataMaps[i][row * 1920 + col] == 1){
						final_result.at<Vec3b>(row, col)[0] = 255;
						final_result.at<Vec3b>(row, col)[1] = 255;
						final_result.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}	

			//���� ��� ȭ�� ����
			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(kFrameNumber) + "_" + to_string(i) + "_" + date + ".jpg";
			imwrite(resultFileName.c_str(), final_result);

			/**/printf("%d��° ������, %d��° BodyIndex GraphCut �Ϸ�\n", kFrameNumber, i);		
		}
	}

	//�޸� ����
	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (skeletonDataMaps[i] != NULL)
			delete[] skeletonDataMaps[i];
	}
	if (skeletonDataMaps != NULL)
		delete[] skeletonDataMaps;
	
	delete[] skeletonMaps;

	delete[] GC_Masks;

	//TEST: �ҿ� �ð� ����(end)///////////////
	end = clock();
	cout << "����ð� : " << ((end - begin)) << endl;
	//////////////////////////////////////////

	//���α׷� ����
	return 0;
}




BYTE* loadBodyIndexFile(int frameNumber){
	/*
	(1920 * 2) * 1080�� ����
	L1, R1, L2, R2, L3, R3, L4, R4 ... ������ ����
	L�� ���� ������ ��. ��, bodyIndex�� 0~5�������� ��ȣ�� �������
	R�� ���� �ش� ���� ��Ȯ���� ��Ÿ��. �� 0, 1 �� ���� ������ 0�̸� ������ ��, 1�̸� ��Ȯ�� ���� ��Ÿ����.	
	*/
	BinaryReader br(filePath::getInstance()->getBodyIndexPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)3840 * 1080;

	//HACK: ����Ʈ �����͸� ����ϴ� ��, ����� ������Ʈ�� �۾Ƽ� main ���������� delete�� ���� �޸� ������ ���� ��
	BYTE* bodyIndexData = new BYTE[3840 * 1080];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		bodyIndexData[arr_index] = br.ReadBYTE();

		arr_index++;
		cur_pos += sizeof(BYTE);
	}

	return bodyIndexData;

	//bodyIndex�� ����ϱ� ���� ���� �Լ�
	/*
	Mat bodyIndex(con_img.rows, con_img.cols, CV_8UC1, Scalar(255));
	for (int row = 0; row < con_img.rows; row++){
	for (int col = 0; col < con_img.cols; col++){
	if (bodyIndexData[row * 3840 + (col * 2)] == 3)
	bodyIndex.at<uchar>(row, col) = bodyIndexData[row * 3840 + (col * 2) ];
	}
	}

	imwrite("bodyIndex.jpg", bodyIndex);
	*/
}

short* loadMappingFile(int frameNumber){
	/*
	(512 * 2) * 424�� ����
	Y1, X1, Y2, X2, Y3, X3, Y4, X4 ... ������ ����
	���� ����ִ� Y�� X�� ��ǥ ���� 512 * 424�� �ִ� depth�� ���� ��ǥ������
	1920 * 1080���� mapping �� �� ����Ǵ� x, y��ǥ�� ���� ���� ����
	1920 * 1080�� ����� ���� ����ִ� ���(����� ���� �Ұ� ��)�� ������ üũ�� �ʿ���
	*/
	BinaryReader br(filePath::getInstance()->getMappPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)1024 * 424;

	//HACK: ����Ʈ �����͸� ����ϴ� ��, ����� createSkeletonData ���������� delete�� ���� �޸� ������ ���� ��
	short* mappData = new short[1024 * 424];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		mappData[arr_index] = br.ReadInt16();
		arr_index++;
		cur_pos++;
	}

	return mappData;
}

BYTE* loadDepthImageFile(int frameNumber){
	/*
	loadBodyIndexFile�� ������ ���
	��, depth �������� ��� 1byte�� �ƴ϶� 2byte�̹Ƿ� 2byte �� �о�´�.
	*/
	const ushort minDepth = 500;	//�ʹ� ���� depth���� �ɷ����� ���� ��
	const ushort maxDepth = 65535;	//unsigned short's max value
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(filePath::getInstance()->getDepthPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)512 * 424;

	//HACK: ����Ʈ �����͸� ����ϴ� ��, ����� ������Ʈ�� �۾Ƽ� main ���������� delete�� ���� �޸� ������ ���� ��
	BYTE* depthData = new BYTE[512 * 424];

	int arr_index = 0;
	while (cur_pos < file_length)
	{		
		ushort depthValue = br.ReadInt16();		
		depthData[arr_index] = (depthValue >= minDepth && depthValue <= maxDepth ? (depthValue / MapDepthToByte) : 0);
		
		arr_index++;
		cur_pos++;
	}

	return depthData;
}

Point2i** loadSkeletonFile(int frameNumber){
	/*
	���̷��� ������ .csv�κ��� �о�´�.
	�̶� �ػ󵵴� 512x424����(depth)
	���Ŀ� mapping�� ���� 1920x1080�� ���� �� ������ �ؾ���
	*/

	//���� ����, �޸� �Ҵ�, �ʱ�ȭ
	Point2i** skeletonDatas_origin = new Point2i*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		//kJointFromKinectV2�� kinect2���� �����ϴ� joints�� ����(25��)
		skeletonDatas_origin[i] = new Point2i[kJointFromKinectV2];

		for (int j = 0; j < kJointFromKinectV2; j++){
			(skeletonDatas_origin[i][j]).x = 0;
			(skeletonDatas_origin[i][j]).y = 0;
		}
	}

	//���� ��� ����
	FILE* stream = fopen(filePath::getInstance()->getSkeletonPath(), "r");

	char line[1024];
	const int currentPos = 150 * frameNumber;
	int jointsCount = 0;
	int bodyIndex = 0;

	//kinect�� �� 6���� bodyIndex�� ���� �� ����. �� ��° bodyIndex ������ �����ϴ��� Ȯ���ϱ� ���� ����
	for (int i = 0; i < kTotal_BodyIndex; i++){
		isBody[i] = false;
	}

	/*
	HACK : csv���Ͽ� å����ó�� ������ �� ���� ������, �� frame���� �ش� frame����ŭ�� ������ �о�� ��
	�� ���� ���� ����
	ù��° ���� �� ù �����ӿ� ù��° ����� ���� ���
	�ι��� ���� �� ��찡 �ƴ� ���
	*/
	if (frameNumber != 0)
	{
		while (fgets(line, 1024, stream))
		{
			if (currentPos == jointsCount + 1) break;
			jointsCount++;
		}
	}

	/*
	�� �ݺ����� ���� ���� stream�� csv���� ���� frameNumber�� ��ġ�� �̵���
	*/

	jointsCount = 0;

	while (fgets(line, 1024, stream))
	{
		//joints ������ŭ ������(�� ���δ� �ϳ��� joint�� ���� ����) �Ѹ��� ���� ���̹Ƿ�, ���� ����� �б� ���� ���� �ʱ�ȭ
		if (jointsCount == kJointFromKinectV2){
			jointsCount = 0;
			
			bodyIndex++;
		}
		//6���� bodyIndex�� ��� �� ������ ���� frame�� skeleton ������ �� ���� ���� �ǹǷ� �ݺ����� �������´�.
		if (bodyIndex == 6) break;

		/*
		csv������ �д� �κ�
		csv���� ������ x , y , z , state(���� �ŷڵ�) , bodyIndex ���̴�.
		*/
		char* tmp_x = _strdup(line);
		int csv_x = atoi(getfield(tmp_x, 1));
		free(tmp_x);
		char* tmp_y = _strdup(line);
		int csv_y = atoi(getfield(tmp_y, 2));
		free(tmp_y);
		char* tmp_z = _strdup(line);
		int csv_z = atoi(getfield(tmp_z, 3));
		free(tmp_z);
		char* tmp_state = _strdup(line);
		int csv_state = atoi(getfield(tmp_state, 4));
		free(tmp_state);
		char* tmp_bodyIndex = _strdup(line);
		int csv_bodyIndex = atoi(getfield(tmp_bodyIndex, 5));
		free(tmp_bodyIndex);

		/*
		���� bodyIndex������ 9999��� ���� �ش� ��ȣ�� ����� �������� �ʾҴٴ� �Ҹ��̹Ƿ� 
		body�� ���ٰ� isBody������ ����
		*/
		if (csv_bodyIndex == 9999)
		{
			isBody[bodyIndex] = false;
			/*
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (bodyIndex == i)
					isBody[i] = false;
			}
			*/
		}
		if (csv_bodyIndex != 9999)
		{
			isBody[bodyIndex] = true;
			/*
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (bodyIndex == i)
					isBody[i] = true;
			}
			*/
		}

		//���� joint��ȣ�� x, y ��ǥ ����
		(skeletonDatas_origin[bodyIndex][jointsCount]).x = csv_x;
		(skeletonDatas_origin[bodyIndex][jointsCount]).y = csv_y;
		jointsCount++;
	}

	fclose(stream);
	return skeletonDatas_origin;
}



void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData){
	/*
	BodyIndex�� Seed�� �Է��� �� kinect���� �����ϴ� BodyIndex�� 512 * 424, �� depth �ػ� ����
	�� ���������� 1920 * 1080 ���󿡼��� segmentation�̱� ������ �� �ػ� ������ �� HR_BodyIndex�� �̿�
	���� �ʿ������� interpolation�� ������ �� ���ۿ� ����, ���� kinect���� �����ϴ� ������ interpolation�� ������ ������ �ʿ䰡 ����
	���� HR_BodyIndex�� �ػ󵵴� (1920 * 2) * 1080�� �Ǹ� ù��° ���� bodyIndex��, �ι�° ���� ���� Ű��Ʈ ��(1) Ȥ�� interpolation�� ��(0)���� ������ ����.
	*/
	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH * 2; col += 2){

			if (bodyIndexData[row * IMAGE_WIDTH * 2 + col + 1] == 1){
				int index = bodyIndexData[row * IMAGE_WIDTH * 2 + col];
				if (index <= kTotal_BodyIndex)
					mask[index].at<uchar>(row, (col / 2)) = GC_FGD;
			}
			else{//bodyIndexData[row * IMAGE_WIDTH * 2 + col + 1] == 0
				int index = bodyIndexData[row * IMAGE_WIDTH * 2 + col];
				if (index <= kTotal_BodyIndex)
					mask[index].at<uchar>(row, (col / 2)) = GC_PR_FGD;
			}
		}
	}

}

void addSeedBySkeleton(Mat* mask, BYTE** skeletonDatas){
	/*
	Skeleton joint�� ���� ���� Skeleton Lines�� Seed�� �Է�
	*/
	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDatas[i][row * IMAGE_WIDTH + col] == 1){
					mask[i].at<uchar>(row, col) = GC_FGD;
				}
			}
		}
	}

}



BYTE** createSkeletonData(int frameNumber, Point2i** skeletonDatas_origin){
	/*
	skeletonDatas_origin(�ػ� 512x424)�� skeleton joint���� �����κ���
	skeletonDatas_hd(�ػ� 1920x1080)�� ������ ����鼭, �� joints���� �ٷ� �����Ѵ�.
	*/
	//�޸� �Ҵ� �� �ʱ�ȭ
	BYTE** skeletonDatasMap = new BYTE*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonDatasMap[i] = new BYTE[IMAGE_HEIGHT * IMAGE_WIDTH];
	}		
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			for (int i = 0; i < kTotal_BodyIndex; i++){
				skeletonDatasMap[i][row * 1920 + col] = 0;
			}			
		}
	}

	//Mapping ������ ��� �ִ� Data�ε�
	short* mappingDats = loadMappingFile(frameNumber);

	for (int i = 0; i < kTotal_BodyIndex; i++){

		if (isBody[i]){
			Point2i* skeletonDatas_hd = new Point2i[kJointFromKinectV2];
			for (int j = 0; j < kJointFromKinectV2; j++){
				skeletonDatas_hd[j] = mappingLowToHigh(skeletonDatas_origin[i][j], mappingDats);
				printf("%d = %d %d\n", i, skeletonDatas_hd[j].x, skeletonDatas_hd[j].y);
			}

			CreateSkeletonLines(skeletonDatas_hd, IMAGE_WIDTH, IMAGE_HEIGHT, 1, skeletonDatasMap[i]);			
			delete[] skeletonDatas_hd;
		}
	}

	delete[] mappingDats;
	return skeletonDatasMap;
}

Mat* createSkeletonWeightMap(int frame, BYTE** skeletonDatas){
	/*
	Skeleton Lines�� �̿��Ͽ� �ش� �ȼ� �ֺ��� ����þ� �Լ��� ���� �Ÿ��� �־��� ���� weight ���� ���� �������� weight map ����
	�Ʒ� ������ ����þ� �Լ��� �⺻ �Լ��� ������ ���������,
	x������ �ѹ� y������ �ѹ� �� �� ����Ѵ�.
	sigma�� ����þ� �Լ��� ������ ������ �����ϸ�, range�� �ȼ� ������ ��� ������ ������ �ȼ����� weight�� ���� ���� �����Ѵ�.
	*/
	Mat* skeletonMaps = new Mat[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonMaps[i] = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	}


	double sigma = 20.0;
	const int range = 100;

	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){

			for (int i = 0; i < kTotal_BodyIndex; i++){
				//���� ��ġ�� skeleton ������ ���� ��,
				if (skeletonDatas[i][row * IMAGE_WIDTH + col] == 1){

					//x������ ����
					for (int offset = -1 * range; offset <= range; offset++){
						if (col + offset >= 0 && col + offset < IMAGE_WIDTH){
							double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(offset, 2)) / (2.0 * powf(sigma, 2)))));
							weight *= 500;

							//���� weight�� ���� pixel�� ������ �ִ� weight���� ���� ���� weight�� �����Ѵ�.
							//Skeleton Line�� 1���� �ȼ� ������ �ƴ϶� �β��� �ֱ� ������
							if (weight >= skeletonMaps[i].at<uchar>(row, col + offset))
								skeletonMaps[i].at<uchar>(row, col + offset) = weight;

						}
					}

					//y������ ����
					for (int offset = -1 * range; offset <= range; offset++){
						if (row + offset >= 0 && row + offset < IMAGE_HEIGHT){
							double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(offset, 2)) / (2.0 * powf(sigma, 2)))));
							weight *= 500;

							if (weight >= skeletonMaps[i].at<uchar>(row + offset, col)){}
							skeletonMaps[i].at<uchar>(row + offset, col) = weight;

						}
					}

				}
			}
		}
	}

	for (int i = 0; i < kTotal_BodyIndex; i++){
		std::string skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_" + to_string(i) + ".jpg";
		imwrite(skeletonFilePath.c_str(), skeletonMaps[i]);
	}

	return skeletonMaps;
}

void CreateSkeletonLines(Point2i* jointPoints, int width, int height, int stroke, BYTE* skeletonMap){
	/*
	joint���� �����ϴ� �κ�, joint ������ȣ�� kinectv2���� ����
	�Ӹ�-�� ��
	*/
	//SPINE_BASE - SPINE_MID
	drawLineBy(jointPoints[0], jointPoints[1], width, height, stroke, skeletonMap);
	//SPINE_BASE - HIP_RIGHT
	drawLineBy(jointPoints[0], jointPoints[16], width, height, stroke, skeletonMap);
	//SPINE_BASE - HIP_LEFT
	drawLineBy(jointPoints[0], jointPoints[12], width, height, stroke, skeletonMap);
	//SPINE_MID - SPINE_SHOULDER
	drawLineBy(jointPoints[1], jointPoints[20], width, height, stroke, skeletonMap);
	//SPINE_SHOULDER - NECK
	drawLineBy(jointPoints[20], jointPoints[2], width, height, stroke, skeletonMap);
	//SPINE_SHOULDER - SHOULDER_RIGHT
	drawLineBy(jointPoints[20], jointPoints[8], width, height, stroke, skeletonMap);
	//SHOULDER_RIGHT - SHOULDER_LEFT
	drawLineBy(jointPoints[20], jointPoints[4], width, height, stroke, skeletonMap);
	//NECT - HEAD
	drawLineBy(jointPoints[2], jointPoints[3], width, height, stroke, skeletonMap);
	//SHOULDER_LEFT - ELBOW_RIGHT
	drawLineBy(jointPoints[8], jointPoints[9], width, height, stroke, skeletonMap);
	//ELBOW_RIGHT - WRIST_RIGHT
	drawLineBy(jointPoints[9], jointPoints[10], width, height, stroke, skeletonMap);
	//WRIST_RIGHT - HAND_RIGHT
	drawLineBy(jointPoints[10], jointPoints[11], width, height, stroke, skeletonMap);
	//HAND_RIGHT - HAND_TIP_RIGHT
	drawLineBy(jointPoints[11], jointPoints[23], width, height, stroke, skeletonMap);
	//HAND_RIGHT - THUMB_RIGHT
	drawLineBy(jointPoints[11], jointPoints[24], width, height, stroke, skeletonMap);
	//SHOULDER_LEFT - ELBOW_LEFT
	drawLineBy(jointPoints[4], jointPoints[5], width, height, stroke, skeletonMap);
	//ELBOW_LEFT - WRIST_LEFT
	drawLineBy(jointPoints[5], jointPoints[6], width, height, stroke, skeletonMap);
	//WRIST_LEFT - HAND_LEFT
	drawLineBy(jointPoints[6], jointPoints[7], width, height, stroke, skeletonMap);
	//HAND_LEFT - HAND_TIP_LEFT
	drawLineBy(jointPoints[7], jointPoints[21], width, height, stroke, skeletonMap);
	//HAND_LEFT - THUMB_LEFT
	drawLineBy(jointPoints[7], jointPoints[22], width, height, stroke, skeletonMap);
	//HIP_RIGHT - KNEE_RIGHT
	drawLineBy(jointPoints[16], jointPoints[17], width, height, stroke, skeletonMap);
	//KNEE_RIGHT - ANKLE_RIGHT
	drawLineBy(jointPoints[17], jointPoints[18], width, height, stroke, skeletonMap);
	//ANKLE_RIGHT - FOOT_RIGHT
	drawLineBy(jointPoints[18], jointPoints[19], width, height, stroke, skeletonMap);
	//HIP_LEFT - KNEE_LEFT
	drawLineBy(jointPoints[12], jointPoints[13], width, height, stroke, skeletonMap);
	//KNEE_LEFT - ANKLE_LEFT
	drawLineBy(jointPoints[13], jointPoints[14], width, height, stroke, skeletonMap);
	//ANKLE_LEFT - FOOT_LEFT
	drawLineBy(jointPoints[14], jointPoints[15], width, height, stroke, skeletonMap);
}



Point2i mappingLowToHigh(Point2i point, short* mappingData){
	/*
	NOTE: loadMappingFile�Լ� ���� ����
	9999���� kinect���� �߸��� ��(�� ������ ���� ���Ͽ�)�� �־��� �� ������ ��
	�� ���� ������ -1�� ������ x, y���� �����Ͽ� ���Ŀ� �������� �����ϵ��� ����ó�� ��.
	*/
	Point2i pos;
	if (point.x == 9999) pos.x = -1;
	else pos.x = mappingData[point.y * 1024 + (2 * point.x) + 1];

	if (point.y == 9999) pos.y = -1;
	else pos.y = mappingData[point.y * 1024 + (2 * point.x)];

	printf("point = %d %d\n", point.x, point.y);

	if (pos.x < 0 || pos.x >= 1920) pos.x = -1;
	if (pos.y < 0 || pos.y >= 1080) pos.y = -1;


	return pos;
}

void expandPixelBy(int x, int y, int width, int height, int stroke, BYTE* skeletonDatas){
	/*
	stroke������ �ش� ��ġ�� line �β� ����
	*/
	int offset = stroke / 2;
	if (offset < 0) offset = 0;

	for (int off_y = -offset; off_y <= offset; off_y++){
		for (int off_x = -offset; off_x <= offset; off_x++){

			if (off_y + y >= 0 && off_y + y < height){
				if (off_x + x >= 0 && off_x + x < width){
					skeletonDatas[(y + off_y) * width + (x + off_x)] = 1;
				}
			}
		}
	}
}

void drawLineBy(Point2i start, Point2i end, int width, int height, int stroke, BYTE* skeletonDatas){
	/*
	�� ���� ���⸦ ���� ��, ������ �������� ���ϰ� �� �����Ŀ� �ִ� ��� �ȼ����� ĥ�Ͽ� skeleton line�� ����� �Լ�
	start��end�� ���⸦ ���ϰ�, stroke�� line�� �β��� ���Ѵ�.
	*/
	if (start.x == -1 && start.y == -1) return;
	double x1 = start.x;
	double y1 = start.y;

	if (end.x == -1 && end.y == -1) return;
	double x2 = end.x;
	double y2 = end.y;


	int s, e;
	if (x1 == x2){
		if (y1 >= y2){
			s = y2;
			e = y1;
		}
		else {
			s = y1;
			e = y2;
		}

		for (int y = s; y <= e; y++){
			//skeletonDatas[y * IMAGE_WIDTH + (int)x1] = 1;
			expandPixelBy(x1, y, width, height, stroke, skeletonDatas);
		}
	}
	else if (y1 == y2){
		if (x1 >= x2){
			s = x2;
			e = x1;
		}
		else {
			s = x1;
			e = x2;
		}

		for (int x = s; x <= e; x++){
			//skeletonDatas[(int)y1 * IMAGE_WIDTH + x] = 1;
			expandPixelBy(x, y1, width, height, stroke, skeletonDatas);
		}
	}
	else{
		float m = (y2 - y1) / (x2 - x1);

		if (abs(x2 - x1) >= abs(y2 - y1)) {
			if (x1 >= x2){
				s = x2;
				e = x1;
			}
			else {
				s = x1;
				e = x2;
			}
			for (int x = s; x <= e; x++){
				int y = (int)(m * (x - x1) + y1 + 0.5);
				//skeletonDatas[y * IMAGE_WIDTH + x] = 1;
				expandPixelBy(x, y, width, height, stroke, skeletonDatas);
			}
		}
		else{
			if (y1 >= y2){
				s = y2;
				e = y1;
			}
			else {
				s = y1;
				e = y2;
			}
			for (int y = s; y <= e; y++){
				int x = (int)((y - y1) * (1 / m) + x1 + 0.5);
				//skeletonDatas[y * IMAGE_WIDTH + x] = 1;
				expandPixelBy(x, y, width, height, stroke, skeletonDatas);
			}
		}
	}

}

const char* getfield(char* line, int num)
{
	/*
	csv�� ������ �о���� �� ���� �и��ϴ� �Լ�
	*/
	const char* tok;
	for (tok = strtok(line, ",");
		tok && *tok;
		tok = strtok(NULL, ",\n"))
	{
		if (!--num)
			return tok;
	}
	return NULL;
}