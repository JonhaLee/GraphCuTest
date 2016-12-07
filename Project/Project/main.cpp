#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "BinaryStream.h"

#include "gc.h"
#include "filePath.h"


using namespace cv;

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define PIE 3.1419

const static int kTotal_BodyIndex = 6;

BYTE* bodyIndexData;
BYTE* depthData;
short* mappData;
BYTE** skeletonDatas;
Mat skeletonMaps[kTotal_BodyIndex];
bool isBody[kTotal_BodyIndex];

void loadDatas(int index);
void setSkeletonData(int index);
void setSkeletonLine(Point2i start, Point2i end, int index);
void setSkeletonWeightMap(int frame);
Point2i get(int _x, int _y);
void set(int x, int y, int index);

int main(){
	printf("OpenCV Version : %s\n\n", CV_VERSION);
	
	for (int i = 0; i < kTotal_BodyIndex; i++){
		isBody[i] = false;
	}

	//반복
	for (int frameNumber = 0; frameNumber < 100; frameNumber++){
		Mat con_img = imread(filePath::getInstance()->getColorPath(frameNumber));
		Mat result; // 분할 (4자기 가능한 값)
		//GrabCut에 사용되는 변수들
		Mat back, fore;	//모델(초기 사용)
		Rect rect(10, 10, 100, 100);

		printf("%d번째 프레임, Data 로드 시작\n", frameNumber);
		//데이터 로드
		loadDatas(frameNumber);
		printf("%d번째 프레임, Data 로드 완료\n", frameNumber);

		//현재 프레임 스켈레톤 map 설정
		printf("%d번째 프레임, 스켈레톤 Data 설정 시작\n", frameNumber);
		setSkeletonData(frameNumber);
		printf("%d번째 프레임, 스켈레톤 Data 설정 완료\n", frameNumber);

		//bodydata를 seed로 넣기
		printf("%d번째 프레임, BodyData Seed로 넣기 시작\n", frameNumber);

		Mat GC_Mask[kTotal_BodyIndex];
		for (int i = 0; i < kTotal_BodyIndex; i++){
			GC_Mask[i] = Mat(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		}

		for (int row = 0; row < 1080; row++){
			for (int col = 0; col < 3840; col += 2){
				if (bodyIndexData[row * 3840 + col + 1] == 1){
					int index = bodyIndexData[row * 3840 + col];
					if (index <= kTotal_BodyIndex)
						GC_Mask[index].at<uchar>(row, (col / 2)) = GC_FGD;					
				}
				else{
					int index = bodyIndexData[row * 3840 + col];
					if (index <= kTotal_BodyIndex)
						GC_Mask[index].at<uchar>(row, (col / 2)) = GC_PR_FGD;					
				}

			}
		}
		printf("%d번째 프레임, BodyData Seed로 넣기 완료\n", frameNumber);


		//skeletondata를 seed로 넣기
		printf("%d번째 프레임, SkeletonData로 시드 넣기 시작\n", frameNumber);
		for (int row = 0; row < 1080; row++){
			for (int col = 0; col < 1920; col++){
				for (int i = 0; i < kTotal_BodyIndex; i++){
					if (skeletonDatas[i][row * 1920 + col] == 1){
						GC_Mask[i].at<uchar>(row, col) = GC_FGD;
					}
				}				
			}
		}
		printf("%d번째 프레임, SkeletonData로 시드 넣기 완료\n", frameNumber);
		/*
		//배경 시드를 주기 위해서 넣었던 부분
		for (int row = 200; row < 875; row++){
		for (int i = 0; i < 10; i++){
		GC_Mask.at<uchar>(row, 1050 + i) = GC_BGD;
		GC_Mask.at<uchar>(row, 370 + i) = GC_BGD;
		}
		}
		*/

		//현재 SkeletonWiethMap을 만드는 부분
		setSkeletonWeightMap(frameNumber);

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

		//depth를 출력하기 위해 만든 부분
		/*
		Mat depth(424, 512, CV_8UC1, Scalar(0));

		for (int row = 0; row < 424; row++){
		for (int col = 0; col < 512; col++){
		depth.at<uchar>(row,col) = depthData[row * 512 + col];

		}
		}
		imwrite("124125.jpg", depth);
		*/

		//주어진 정보를 토대로 GraphCut을 돌리는 부분

		for (int i = 0; i < kTotal_BodyIndex; i++){
			if (isBody[i]){
				printf("%d번째 프레임, %d번째 BodyIndex GraphCut 시작\n", frameNumber, i);
				my::GraphCut gc;

				gc.graphCut(con_img, //입력영상		
					GC_Mask[i],//분할 마스크
					rect, //전경을 포함하는 직사각형
					skeletonMaps[i],
					back, fore, //모델
					1,//반복횟수
					GC_INIT_WITH_MASK,	//직사각형 사용
					skeletonMaps[i]);


				Mat fgd_result, pr_fgd_result;

				Mat foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				Mat pr_foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


				compare(GC_Mask[i], cv::GC_FGD, fgd_result, cv::CMP_EQ);
				con_img.copyTo(foreground, fgd_result);

				compare(GC_Mask[i], GC_PR_FGD, pr_fgd_result, CMP_EQ);
				con_img.copyTo(pr_foreground, pr_fgd_result);

				con_img.copyTo(final_result, fgd_result);

				imwrite("fgd.jpg", final_result);

				con_img.copyTo(final_result, pr_fgd_result);

				for (int row = 0; row < 1080; row++){
					for (int col = 0; col < 1920; col++){
						if (skeletonDatas[i][row * 1920 + col] == 1){
							final_result.at<Vec3b>(row, col)[0] = 255;
							final_result.at<Vec3b>(row, col)[1] = 255;
							final_result.at<Vec3b>(row, col)[2] = 0;

						}
					}
				}

				//namedWindow("Result");
				//imshow("Result", final_result);

				//namedWindow("Foreground");
				//imshow("Foreground", foreground);


				std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(frameNumber) + to_string(i) +".jpg";
				imwrite(resultFileName.c_str(), final_result);

				printf("%d번째 프레임, %d번째 BodyIndex GraphCut 완료\n", frameNumber, i);
				//waitKey(0);
			}
		}
	}	
	
	delete bodyIndexData;
	delete mappData;

	//메모리 해제
	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (skeletonDatas[i] != NULL)
			delete[] skeletonDatas[i];
	}
	if (skeletonDatas != NULL)
		delete[] skeletonDatas;
	
	return 0;
}

void setSkeletonWeightMap(int frame){
	
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonMaps[i] = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	}


	double sigma = 20.0;
	const int range = 100;

	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDatas[i][row * IMAGE_WIDTH + col] == 1){					
					for (int offset = -1 * range; offset <= range; offset++){
						if (col + offset >= 0 && col + offset < IMAGE_WIDTH){
							double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(offset, 2)) / (2.0 * powf(sigma, 2)))));

							//printf("%f ", weight);
							if (weight >= skeletonMaps[i].at<uchar>(row, col + offset))
								skeletonMaps[i].at<uchar>(row, col + offset) = weight * 500;

						}
					}					
					for (int offset = -1 * range; offset <= range; offset++){
						if (row + offset >= 0 && row + offset < IMAGE_HEIGHT){
							double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(offset, 2)) / (2.0 * powf(sigma, 2)))));

							//printf("%f ", weight);
							if (weight >= skeletonMaps[i].at<uchar>(row + offset, col)){}
							skeletonMaps[i].at<uchar>(row + offset, col) = weight * 500;

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

	printf("Weight Map 구함\n");
}

void loadDatas(int index){
	delete bodyIndexData;
	delete mappData;
	delete depthData;

	BinaryReader br(filePath::getInstance()->getBodyIndexPath(index));
	int pos = 0;
	int length = (int)3840 * 1080;

	bodyIndexData = new BYTE[3840 * 1080];
	
	int count = 0;
	while (pos < length)
	{
		bodyIndexData[count] = br.ReadBYTE();		

		count++;
		pos += sizeof(BYTE);
	}

	BinaryReader br2(filePath::getInstance()->getMappPath(index));
	pos = 0;
	length = (int)1024 * 424;

	mappData = new short[1024 * 424];

	count = 0;
	while (pos < length)
	{
		mappData[count] = br2.ReadInt16();
		count++;
		pos ++;
	}

	BinaryReader br3(filePath::getInstance()->getDepthPath(index));
	pos = 0;
	length = (int)512 * 424;

	depthData = new BYTE[512 * 424];

	count = 0;
	while (pos < length)
	{
		depthData[count] = br3.ReadBYTE();

		count++;
		pos += sizeof(BYTE);
	}
}

const char* getfield(char* line, int num)
{
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


void setSkeletonData(int frame){
	//메모리 할당
	skeletonDatas = new BYTE*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonDatas[i] = new BYTE[IMAGE_HEIGHT * IMAGE_WIDTH];
	}	
	
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			for (int i = 0; i < kTotal_BodyIndex; i++){
				skeletonDatas[i][row * 1920 + col] = 0;
			}			
		}
	}

	Point2i* skeletonPoints = new Point2i[25];

	FILE* stream = fopen(filePath::getInstance()->getSkeletonPath(), "r");

	char line[1024];
	const int currentPos = 150 * frame;
	int count = 0;

	if (frame != 0)
	{
		while (fgets(line, 1024, stream))
		{
			if (currentPos == count + 1) break;
			count++;
		}
	}

	count = 0;
	int index = 0;

	while (fgets(line, 1024, stream))
	{		
		if (count == 25){
			count = 0;
		
			//SPINE_BASE - SPINE_MID
			setSkeletonLine(skeletonPoints[0], skeletonPoints[1], index);
			//SPINE_BASE - HIP_RIGHT
			setSkeletonLine(skeletonPoints[0], skeletonPoints[16], index);
			//SPINE_BASE - HIP_LEFT
			setSkeletonLine(skeletonPoints[0], skeletonPoints[12], index);
			//SPINE_MID - SPINE_SHOULDER
			setSkeletonLine(skeletonPoints[1], skeletonPoints[20], index);
			//SPINE_SHOULDER - NECK
			setSkeletonLine(skeletonPoints[20], skeletonPoints[2], index);
			//SPINE_SHOULDER - SHOULDER_RIGHT
			setSkeletonLine(skeletonPoints[20], skeletonPoints[8], index);
			//SHOULDER_RIGHT - SHOULDER_LEFT
			setSkeletonLine(skeletonPoints[20], skeletonPoints[4], index);
			//NECT - HEAD
			setSkeletonLine(skeletonPoints[2], skeletonPoints[3], index);
			//SHOULDER_LEFT - ELBOW_RIGHT
			setSkeletonLine(skeletonPoints[8], skeletonPoints[9], index);
			//ELBOW_RIGHT - WRIST_RIGHT
			setSkeletonLine(skeletonPoints[9], skeletonPoints[10], index);
			//WRIST_RIGHT - HAND_RIGHT
			setSkeletonLine(skeletonPoints[10], skeletonPoints[11], index);
			//HAND_RIGHT - HAND_TIP_RIGHT
			setSkeletonLine(skeletonPoints[11], skeletonPoints[23], index);
			//HAND_RIGHT - THUMB_RIGHT
			setSkeletonLine(skeletonPoints[11], skeletonPoints[24], index);
			//SHOULDER_LEFT - ELBOW_LEFT
			setSkeletonLine(skeletonPoints[4], skeletonPoints[5], index);
			//ELBOW_LEFT - WRIST_LEFT
			setSkeletonLine(skeletonPoints[5], skeletonPoints[6], index);
			//WRIST_LEFT - HAND_LEFT
			setSkeletonLine(skeletonPoints[6], skeletonPoints[7], index);
			//HAND_LEFT - HAND_TIP_LEFT
			setSkeletonLine(skeletonPoints[7], skeletonPoints[21], index);
			//HAND_LEFT - THUMB_LEFT
			setSkeletonLine(skeletonPoints[7], skeletonPoints[22], index);
			//HIP_RIGHT - KNEE_RIGHT
			setSkeletonLine(skeletonPoints[16], skeletonPoints[17], index);
			//KNEE_RIGHT - ANKLE_RIGHT
			setSkeletonLine(skeletonPoints[17], skeletonPoints[18], index);
			//ANKLE_RIGHT - FOOT_RIGHT
			setSkeletonLine(skeletonPoints[18], skeletonPoints[19], index);
			//HIP_LEFT - KNEE_LEFT
			setSkeletonLine(skeletonPoints[12], skeletonPoints[13], index);
			//KNEE_LEFT - ANKLE_LEFT
			setSkeletonLine(skeletonPoints[13], skeletonPoints[14], index);
			//ANKLE_LEFT - FOOT_LEFT
			setSkeletonLine(skeletonPoints[14], skeletonPoints[15], index);

			index++;

			for (int i = 0; i < 25; i++){
				skeletonPoints[i].x = 0;
				skeletonPoints[i].y = 0;
			}
		}
		if (index == 6) break;
		
		char* tmp_x = _strdup(line);
		int x = atoi(getfield(tmp_x, 1));
		free(tmp_x);
		char* tmp_y = _strdup(line);
		int y = atoi(getfield(tmp_y, 2));
		free(tmp_y);
		char* tmp_z = _strdup(line);
		int z = atoi(getfield(tmp_z, 3));		
		free(tmp_z);
		char* tmp_state = _strdup(line);
		int state = atoi(getfield(tmp_state, 4));
		free(tmp_state);
		char* tmp_bodyIndex = _strdup(line);
		int bodyIndex = atoi(getfield(tmp_bodyIndex, 5));
		free(tmp_bodyIndex);
		
		if (bodyIndex == 9999)
		{
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (index == i)
					isBody[i] = false;
			}			
		}
		if (bodyIndex != 9999)
		{
			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (index == i)
					isBody[i] = true;
			}
		}
	
		skeletonPoints[count] = get(x, y);
		count++;	
	}
	
	fclose(stream);	
}

void setSkeletonLine(Point2i start, Point2i end, int index){
	double x1 = start.x;
	double y1 = start.y;
	double x2 = end.x;
	double y2 = end.y;

	if (x1 == -1 && y1 == -1) return;
	if (x2 == -1 && y2 == -1) return;


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
			set(x1, y, index);
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
			set(x, y1, index);
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
				set(x, y, index);
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
				set(x, y, index);
			}
		}
	}

}
Point2i get(int _x, int _y){
	Point2i pos;
	if (_x == 9999) pos.x = -1;
	else pos.x = mappData[_y * 1024 + (2 * _x) + 1];
	if (_y == 9999) pos.y = -1;
	else pos.y = mappData[_y * 1024 + (2 * _x)];

	if (pos.x < 0 || pos.x >= 1920) pos.x = -1;
	if (pos.y < 0 || pos.y >= 1080) pos.y = -1;

	return pos;
}
void set(int x, int y, int index){

	for (int off_y = -3; off_y <= 3; off_y++){
		for (int off_x = -3; off_x <= 3; off_x++){
			if (off_y + y >= 0 && off_y + y < 1080){
				if (off_x + x >= 0 && off_x + x < 1920){
					skeletonDatas[index][(y + off_y) * 1920 + (x + off_x)] = 1;					
				}
			}
		}
	}
}