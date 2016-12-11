#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "BinaryStream.h"

#include "gc.h"
#include "filePath.h"

//테스트를 위한 코드
#include <time.h>

using namespace cv;

//상수들
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define PIE 3.1419

const static int kTotal_BodyIndex = 6;
const static int kJointFromKinectV2 = 25;

bool isBody[kTotal_BodyIndex];


//File로부터 Data를 읽어오는 함수들
BYTE* loadBodyIndexFile(int frameNumber);
short* loadMappingFile(int frameNumber);
BYTE* loadDepthImageFile(int frameNumber);
Point2i** loadSkeletonFile(int frameNumber);


//Seed를 입력하는 함수들
void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData);
void addSeedBySkeleton(Mat* mask, BYTE** skeletonData);


//Skeleton 정보 가공 함수들
BYTE** createSkeletonData(int frameNumber, Point2i** skeletonDatas_origin);
Mat* createSkeletonWeightMap(int frame, BYTE** skeletonDatas);
void CreateSkeletonLines(Point2i* jointPoints, int width, int height, int stroke, BYTE* skeletonMap);


//유틸리티 함수들
Point2i mappingLowToHigh(Point2i point, short* mappingData);
void expandPixelBy(int x, int y, int width, int height, int stroke, BYTE* skeletonDatas);
void drawLineBy(Point2i start, Point2i end, int width, int height, int stroke, BYTE* skeletonDatas);

const char* getfield(char* line, int num);



int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	//TEST: 소요시간 측정(begin)//////////////
	clock_t begin, end;
	begin = clock();
	//////////////////////////////////////////


	//처리하고자 하는 영상의 frame을 입력(반복을 대비해 따로 빼놓음)
	const int kFrameNumber = 70;


	//BodyIndexData를 얻기 위해 local files을 읽어온다.
	/**/printf("%d번째 프레임, BodyIndexData 로드 시작\n", kFrameNumber);
	BYTE* bodyIndexData = loadBodyIndexFile(kFrameNumber);
	/**/printf("%d번째 프레임, BodyIndexData 로드 완료\n", kFrameNumber);

	/*
	GraphCut에 사용 될 Seed(foreground or background)를 입력
	kinect에서 제공해주는 BodyIndex 개수만큼(현재는 6) GraphCut을 돌리기 위해 각각의 Seed Mask를 생성
	기본적으로 모든 Seed를 PR_Background로 주어준 다음에, 뒤에서 PR_Foreground와 Foreground를 다시 입력
	*/
	/**/printf("%d번째 프레임, BodyIndexData Seed로 넣기 시작\n", kFrameNumber);
	Mat* GC_Masks = new Mat[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		GC_Masks[i] = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(GC_PR_BGD));
	}	
	addSeedByBodyIndex(GC_Masks, bodyIndexData);	
	//메모리 해제
	delete[] bodyIndexData;
	/**/printf("%d번째 프레임, BodyIndexData Seed로 넣기 완료\n", kFrameNumber);


	/*
	스켈레톤 정보 설정부분
	스켈레톤 joint 정보를 얻기 위해 local file(.csv)로부터 정보를 얻어오고
	해당 joint 정보를 바탕으로 뼈대(line)까지 생성하는 부분
	*/	
	/**/printf("%d번째 프레임, 스켈레톤 csv 파일 로드 시작\n", kFrameNumber);
	Point2i** skeletonDatas_origin = loadSkeletonFile(kFrameNumber);
	/**/printf("%d번째 프레임, 스켈레톤 csv 파일 로드 완료\n", kFrameNumber);


	/**/printf("%d번째 프레임, 스켈레톤 Data 생성 시작\n", kFrameNumber);
	BYTE** skeletonDataMaps = createSkeletonData(kFrameNumber, skeletonDatas_origin);
	/**/printf("%d번째 프레임, 스켈레톤 Data 생성 완료\n", kFrameNumber);

		
	/**/printf("%d번째 프레임, SkeletonData로 시드 넣기 시작\n", kFrameNumber);
	addSeedBySkeleton(GC_Masks, skeletonDataMaps);
	/**/printf("%d번째 프레임, SkeletonData로 시드 넣기 완료\n", kFrameNumber);


	//현재 SkeletonWiethMap을 만드는 부분
	/**/printf("%d번째 프레임, SkeletonData로 가중치 맵 생성 시작\n", kFrameNumber);
	Mat* skeletonMaps = createSkeletonWeightMap(kFrameNumber, skeletonDataMaps);
	/**/printf("%d번째 프레임, SkeletonData로 가중치 맵 생성 완료\n", kFrameNumber);
		


	//입력 영상
	Mat con_img = imread(filePath::getInstance()->getColorPath(kFrameNumber));

	
	/*
	필요한 image 파일들을 만드는 부분
	*/

	//이미지 파일 마지막에 날짜와 시간을 집어넣기 위함
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

	//저장을 위한 path
	std::string Spath_inputImage			= filePath::getInstance()->getResultPath() + "input" + to_string(kFrameNumber) + "_" + date + ".jpg";
	std::string Spath_depthNskeletonImage	= filePath::getInstance()->getResultPath() + "depth_skeleton" + to_string(kFrameNumber) + "_" + date + ".jpg";
	std::string Spath_HRskeletonImage = filePath::getInstance()->getResultPath() + "HRskeleton" + to_string(kFrameNumber) + "_" + date + ".jpg";

		
	//깊이 값 + skeleton 처리 부분
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


	//저장을 하는 함수
	imwrite(Spath_inputImage.c_str(), con_img);			//입력 영상
	imwrite(Spath_depthNskeletonImage.c_str(), depthNskeleton);	//깊이 값 + skeleton
	imwrite(Spath_HRskeletonImage.c_str(), HRSkeleton);
	
	//메모리 해제
	delete[] depthData;
	delete[] skeleton_origin_map;




	//주어진 정보를 토대로 GraphCut을 돌리는 부분

	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (isBody[i]){
			/**/printf("%d번째 프레임, %d번째 BodyIndex GraphCut 시작\n", kFrameNumber, i);
			
			//GrabCut에 사용되는 변수들
			//여기서는 GrabCut을 GraphCut처럼 이용하기 때문에 아래 변수들의 리턴값들을 따로 사용하지 않음.
			my::GraphCut gc;
			Mat back, fore;	//모델(초기 사용)
			Rect rect(10, 10, 100, 100);

			//실제로는 GraphCut처럼 작동
			gc.graphCut(con_img,		//입력영상		
				GC_Masks[i],			//분할 마스크, 해당 마스크의 전경과 배경으로 GraphCut 수행
				rect,					//전경을 포함하는 직사각형, 여기서는 MASK 모드를 사용함으로 rect의 값은 전혀 사용하지 않는다.
				back, fore,				//모델
				1,						//반복횟수
				GC_INIT_WITH_MASK,		//Mask를 사용하는 모드 선택
				skeletonMaps[i]);		//Skeleton Weight Map


			/*
			GraphCut의 결과값으로 나오는 GC_Masks에서 원하는 결과값을 뽑아서(compare을 통해) 저장하기 위한 변수
			*/
			Mat fgd_result, pr_fgd_result;

			/*
			결과영상들을 이미지 파일로 저장하기 위한 변수들
			*/
			//Mat foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			//Mat pr_foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));

			//결과 영상에서 GC_FGD인 픽셀들만 fgd_result에 저장
			compare(GC_Masks[i], cv::GC_FGD, fgd_result, cv::CMP_EQ);
			//그 후에 입력 영상과 비교하여 foreground라는 변수에 대응되는 컬러 값을 저장
			//즉, GC_FGD라고 나온 결과값 위치의 컬러 영상만 저장
			//con_img.copyTo(foreground, fgd_result);

			//마찬가지로 GC_PR_FGD에 대해 진행
			compare(GC_Masks[i], GC_PR_FGD, pr_fgd_result, CMP_EQ);
			//con_img.copyTo(pr_foreground, pr_fgd_result);

			//최종 결과 파일(final_result)에 fgd, pr_fgd의 컬러 영상값을 저장
			con_img.copyTo(final_result, fgd_result);
			con_img.copyTo(final_result, pr_fgd_result);

			//final_result 영상과 skeletonData의 값을 합쳐서 컬러 영상 위에 skeleton line이 보이게 표시하는 작업
			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonDataMaps[i][row * 1920 + col] == 1){
						final_result.at<Vec3b>(row, col)[0] = 255;
						final_result.at<Vec3b>(row, col)[1] = 255;
						final_result.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}	

			//최종 결과 화면 저장
			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(kFrameNumber) + "_" + to_string(i) + "_" + date + ".jpg";
			imwrite(resultFileName.c_str(), final_result);

			/**/printf("%d번째 프레임, %d번째 BodyIndex GraphCut 완료\n", kFrameNumber, i);		
		}
	}

	//메모리 해제
	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (skeletonDataMaps[i] != NULL)
			delete[] skeletonDataMaps[i];
	}
	if (skeletonDataMaps != NULL)
		delete[] skeletonDataMaps;
	
	delete[] skeletonMaps;

	delete[] GC_Masks;

	//TEST: 소요 시간 측정(end)///////////////
	end = clock();
	cout << "수행시간 : " << ((end - begin)) << endl;
	//////////////////////////////////////////

	//프로그램 종료
	return 0;
}




BYTE* loadBodyIndexFile(int frameNumber){
	/*
	(1920 * 2) * 1080의 형태
	L1, R1, L2, R2, L3, R3, L4, R4 ... 순으로 저장
	L은 실제 데이터 값. 즉, bodyIndex의 0~5번까지의 번호가 들어있음
	R의 값은 해당 값의 정확도를 나타냄. 즉 0, 1 두 값만 가지며 0이면 보정된 값, 1이면 정확한 값을 나타낸다.	
	*/
	BinaryReader br(filePath::getInstance()->getBodyIndexPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)3840 * 1080;

	//HACK: 스마트 포인터를 고려하는 중, 현재는 프로젝트가 작아서 main 마지막에서 delete를 통해 메모리 해제를 진행 중
	BYTE* bodyIndexData = new BYTE[3840 * 1080];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		bodyIndexData[arr_index] = br.ReadBYTE();

		arr_index++;
		cur_pos += sizeof(BYTE);
	}

	return bodyIndexData;

	//bodyIndex를 출력하기 위해 만든 함수
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
	(512 * 2) * 424의 형태
	Y1, X1, Y2, X2, Y3, X3, Y4, X4 ... 순으로 저장
	여기 들어있는 Y와 X의 좌표 값은 512 * 424에 있던 depth값 기준 좌표값들을
	1920 * 1080으로 mapping 할 때 연결되는 x, y좌표의 실제 값을 저장
	1920 * 1080을 벗어나는 값이 들어있는 경우(노이즈나 측정 불가 등)가 있으니 체크가 필요함
	*/
	BinaryReader br(filePath::getInstance()->getMappPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)1024 * 424;

	//HACK: 스마트 포인터를 고려하는 중, 현재는 createSkeletonData 마지막에서 delete를 통해 메모리 해제를 진행 중
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
	loadBodyIndexFile와 동일한 방식
	단, depth 데이터의 경우 1byte가 아니라 2byte이므로 2byte 씩 읽어온다.
	*/
	const ushort minDepth = 500;	//너무 낮은 depth값은 걸러내기 위한 값
	const ushort maxDepth = 65535;	//unsigned short's max value
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(filePath::getInstance()->getDepthPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)512 * 424;

	//HACK: 스마트 포인터를 고려하는 중, 현재는 프로젝트가 작아서 main 마지막에서 delete를 통해 메모리 해제를 진행 중
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
	스켈레톤 정보를 .csv로부터 읽어온다.
	이때 해상도는 512x424기준(depth)
	차후에 mapping을 통해 1920x1080에 맞춰 값 변경을 해야함
	*/

	//변수 선언, 메모리 할당, 초기화
	Point2i** skeletonDatas_origin = new Point2i*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		//kJointFromKinectV2는 kinect2에서 제공하는 joints의 개수(25개)
		skeletonDatas_origin[i] = new Point2i[kJointFromKinectV2];

		for (int j = 0; j < kJointFromKinectV2; j++){
			(skeletonDatas_origin[i][j]).x = 0;
			(skeletonDatas_origin[i][j]).y = 0;
		}
	}

	//파일 경로 지정
	FILE* stream = fopen(filePath::getInstance()->getSkeletonPath(), "r");

	char line[1024];
	const int currentPos = 150 * frameNumber;
	int jointsCount = 0;
	int bodyIndex = 0;

	//kinect는 총 6명의 bodyIndex를 읽을 수 있음. 몇 번째 bodyIndex 정보가 존재하는지 확인하기 위한 변수
	for (int i = 0; i < kTotal_BodyIndex; i++){
		isBody[i] = false;
	}

	/*
	HACK : csv파일에 책갈피처럼 저장할 수 없기 때문에, 매 frame마다 해당 frame수만큼의 라인을 읽어야 함
	두 가지 경우로 한정
	첫번째 경우는 맨 첫 프레임에 첫번째 사람이 잡힌 경우
	두번재 경우는 위 경우가 아닌 경우
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
	위 반복문을 통해 이제 stream은 csv에서 현재 frameNumber에 위치로 이동함
	*/

	jointsCount = 0;

	while (fgets(line, 1024, stream))
	{
		//joints 개수만큼 읽으면(한 라인당 하나의 joint에 대한 정보) 한명을 읽을 것이므로, 다음 사람을 읽기 위해 변수 초기화
		if (jointsCount == kJointFromKinectV2){
			jointsCount = 0;
			
			bodyIndex++;
		}
		//6명의 bodyIndex을 모두 다 읽으면 현재 frame의 skeleton 정보를 다 읽은 것이 되므로 반복문을 빠져나온다.
		if (bodyIndex == 6) break;

		/*
		csv파일을 읽는 부분
		csv파일 구성은 x , y , z , state(측정 신뢰도) , bodyIndex 순이다.
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
		읽은 bodyIndex정보가 9999라는 것은 해당 번호의 사람이 측정되지 않았다는 소리이므로 
		body가 없다고 isBody변수에 저장
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

		//현재 joint번호의 x, y 좌표 저장
		(skeletonDatas_origin[bodyIndex][jointsCount]).x = csv_x;
		(skeletonDatas_origin[bodyIndex][jointsCount]).y = csv_y;
		jointsCount++;
	}

	fclose(stream);
	return skeletonDatas_origin;
}



void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData){
	/*
	BodyIndex를 Seed를 입력할 때 kinect에서 제공하는 BodyIndex는 512 * 424, 즉 depth 해상도 기준
	본 연구에서는 1920 * 1080 영상에서의 segmentation이기 때문에 두 해상도 매핑을 한 HR_BodyIndex를 이용
	따라서 필연적으로 interpolation한 정보가 들어갈 수밖에 없고, 원래 kinect에서 제공하는 정보와 interpolation한 정보를 구별할 필요가 있음
	따라서 HR_BodyIndex의 해상도는 (1920 * 2) * 1080이 되며 첫번째 값은 bodyIndex값, 두번째 값은 본래 키넥트 값(1) 혹은 interpolation한 값(0)으로 쌍으로 존재.
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
	Skeleton joint를 토대로 만든 Skeleton Lines을 Seed로 입력
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
	skeletonDatas_origin(해상도 512x424)의 skeleton joint들의 값으로부터
	skeletonDatas_hd(해상도 1920x1080)의 정보로 만들면서, 각 joints들을 줄로 연결한다.
	*/
	//메모리 할당 및 초기화
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

	//Mapping 정보를 담고 있는 Data로드
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
	Skeleton Lines을 이용하여 해당 픽셀 주변에 가우시안 함수를 통해 거리가 멀어질 수록 weight 값이 점점 낮아지는 weight map 생성
	아래 수식은 가우시안 함수의 기본 함수를 가져다 사용했으며,
	x축으로 한번 y축으로 한번 두 번 사용한다.
	sigma는 가우시안 함수가 퍼지는 정도를 조절하며, range는 픽셀 상으로 어느 정도의 범위의 픽셀까지 weight을 받을 지를 결정한다.
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
				//현재 위치에 skeleton 정보가 있을 때,
				if (skeletonDatas[i][row * IMAGE_WIDTH + col] == 1){

					//x축으로 진행
					for (int offset = -1 * range; offset <= range; offset++){
						if (col + offset >= 0 && col + offset < IMAGE_WIDTH){
							double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(offset, 2)) / (2.0 * powf(sigma, 2)))));
							weight *= 500;

							//계산된 weight가 현재 pixel이 가지고 있는 weight보다 높을 때만 weight를 설정한다.
							//Skeleton Line은 1개의 픽세 라인이 아니라 두께가 있기 때문에
							if (weight >= skeletonMaps[i].at<uchar>(row, col + offset))
								skeletonMaps[i].at<uchar>(row, col + offset) = weight;

						}
					}

					//y축으로 진행
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
	joint별로 연결하는 부분, joint 고유번호는 kinectv2문서 참고
	머리-목 등
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
	NOTE: loadMappingFile함수 설명 참고
	9999값은 kinect에서 잘못된 값(즉 추적을 하지 못하여)을 주었을 때 나오는 값
	이 값일 때에는 -1의 값으로 x, y값을 저장하여 이후에 과정에서 제외하도록 예외처리 함.
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
	stroke값으로 해당 위치의 line 두께 조절
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
	두 점의 기울기를 구한 뒤, 직선의 방정식을 구하고 그 방정식에 있는 모든 픽셀들을 칠하여 skeleton line을 만드는 함수
	start와end로 기울기를 구하고, stroke로 line의 두께를 정한다.
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
	csv을 한줄을 읽어오면 그 값을 분리하는 함수
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