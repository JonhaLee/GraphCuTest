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

BYTE* bodyIndexData;
BYTE* depthData;
short* mappData;
BYTE* skeletonData0;
BYTE* skeletonData1;
BYTE* skeletonData2;
BYTE* skeletonData3;
BYTE* skeletonData4;
BYTE* skeletonData5;
Mat skeletonMap0;
Mat skeletonMap1;
Mat skeletonMap2;
Mat skeletonMap3;
Mat skeletonMap4;
Mat skeletonMap5;

bool isBody0;
bool isBody1;
bool isBody2;
bool isBody3;
bool isBody4;
bool isBody5;

static int fileNumber = 0;

enum 
{
	none = 0,
	far = 1,
	middle_fal = 2,
	middle =  5,
	middle_near = 7,
	near = 9,
	origin = 10,
};


void loadDatas(int index);
void setSkeletonData(int index);
void setSkeletonLine(Point2i start, Point2i end, int index);
void setSkeletonWeightMap(int frame);
Point2i get(int _x, int _y);
void set(int x, int y, int index);


int main(){
	printf("OpenCV Version : %s\n\n", CV_VERSION);


	isBody0 = false;
	isBody1 = false;
	isBody2 = false;
	isBody3 = false;
	isBody4 = false;
	isBody5 = false;


	

	//반복

	for (int i = 0; i < 100; i++){
		Mat con_img = imread(filePath::getInstance()->getColorPath(i));
		Mat result; // 분할 (4자기 가능한 값)
		//GrabCut에 사용되는 변수들
		Mat back, fore;	//모델(초기 사용)
		Rect rect(10, 10, 100, 100);

		printf("%d번째 프레임, Data 로드 시작\n", i);
		//데이터 로드
		loadDatas(i);
		printf("%d번째 프레임, Data 로드 완료\n", i);

		//현재 프레임 스켈레톤 map 설정
		printf("%d번째 프레임, 스켈레톤 Data 설정 시작\n", i);		
		setSkeletonData(i);
		printf("%d번째 프레임, 스켈레톤 Data 설정 완료\n", i);
		


		//bodydata를 seed로 넣기
		printf("%d번째 프레임, BodyData Seed로 넣기 시작\n", i);
		
		Mat GC_Mask0(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		Mat GC_Mask1(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		Mat GC_Mask2(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		Mat GC_Mask3(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		Mat GC_Mask4(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
		Mat GC_Mask5(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));

		for (int row = 0; row < 1080; row++){
			for (int col = 0; col < 3840; col += 2){				
				if (bodyIndexData[row * 3840 + col + 1] == 1){
					if (bodyIndexData[row * 3840 + col] == 0)
						GC_Mask0.at<uchar>(row, (col / 2)) = GC_FGD;
					else if (bodyIndexData[row * 3840 + col] == 1)
						GC_Mask1.at<uchar>(row, (col / 2)) = GC_FGD;
					else if (bodyIndexData[row * 3840 + col] == 2)
						GC_Mask2.at<uchar>(row, (col / 2)) = GC_FGD;
					else if (bodyIndexData[row * 3840 + col] == 3)
						GC_Mask3.at<uchar>(row, (col / 2)) = GC_FGD;
					else if (bodyIndexData[row * 3840 + col] == 4)
						GC_Mask4.at<uchar>(row, (col / 2)) = GC_FGD;
					else if (bodyIndexData[row * 3840 + col] == 5)
						GC_Mask5.at<uchar>(row, (col / 2)) = GC_FGD;
					else{}						
				}
				else{
					if (bodyIndexData[row * 3840 + col] == 0)
						GC_Mask0.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else if (bodyIndexData[row * 3840 + col] == 1)
						GC_Mask1.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else if (bodyIndexData[row * 3840 + col] == 2)
						GC_Mask2.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else if (bodyIndexData[row * 3840 + col] == 3)
						GC_Mask3.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else if (bodyIndexData[row * 3840 + col] == 4)
						GC_Mask4.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else if (bodyIndexData[row * 3840 + col] == 5)
						GC_Mask5.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					else {}
					
					
					//GC_Mask3.at<uchar>(row, (col / 2)) = GC_PR_FGD;
				}
				
			}
		}
		printf("%d번째 프레임, BodyData Seed로 넣기 완료\n", i);


		//skeletondata를 seed로 넣기
		printf("%d번째 프레임, SkeletonData로 시드 넣기 시작\n", i);
		for (int row = 0; row < 1080; row++){
			for (int col = 0; col < 1920; col++){
				if (skeletonData0[row * 1920 + col] == 1){
					GC_Mask0.at<uchar>(row, col) = GC_FGD;
				}
				if (skeletonData1[row * 1920 + col] == 1){
					GC_Mask1.at<uchar>(row, col) = GC_FGD;
				}
				if (skeletonData2[row * 1920 + col] == 1){
					GC_Mask2.at<uchar>(row, col) = GC_FGD;
				}
				if (skeletonData3[row * 1920 + col] == 1){
					GC_Mask3.at<uchar>(row, col) = GC_FGD;
				}
				if (skeletonData4[row * 1920 + col] == 1){
					GC_Mask4.at<uchar>(row, col) = GC_FGD;
				}
				if (skeletonData5[row * 1920 + col] == 1){
					GC_Mask5.at<uchar>(row, col) = GC_FGD;
				}
			}
		}
		printf("%d번째 프레임, SkeletonData로 시드 넣기 완료\n", i);
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
		setSkeletonWeightMap(i);

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

		
		if (isBody0){
			printf("%d번째 프레임, 0번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc;

			gc.graphCut(con_img, //입력영상		
				GC_Mask0,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap0,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap0);


			Mat fgd_result, pr_fgd_result;

			Mat foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask0, cv::GC_FGD, fgd_result, cv::CMP_EQ);
			con_img.copyTo(foreground, fgd_result);

			compare(GC_Mask0, GC_PR_FGD, pr_fgd_result, CMP_EQ);
			con_img.copyTo(pr_foreground, pr_fgd_result);

			con_img.copyTo(final_result, fgd_result);

			imwrite("fgd.jpg", final_result);

			con_img.copyTo(final_result, pr_fgd_result);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData0[row * 1920 + col] == 1){
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


			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "0.jpg";
			imwrite(resultFileName.c_str(), final_result);

			printf("%d번째 프레임, 0번째 BodyIndex GraphCut 완료\n", i);
			//waitKey(0);
		}

		if (isBody1){
			printf("%d번째 프레임, 1번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc1;

			gc1.graphCut(con_img, //입력영상		
				GC_Mask1,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap1,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap1);


			Mat fgd_result1, pr_fgd_result1;

			Mat foreground1(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground1(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result1(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask1, cv::GC_FGD, fgd_result1, cv::CMP_EQ);
			con_img.copyTo(foreground1, fgd_result1);

			compare(GC_Mask1, GC_PR_FGD, pr_fgd_result1, CMP_EQ);
			con_img.copyTo(pr_foreground1, pr_fgd_result1);

			con_img.copyTo(final_result1, fgd_result1);

			//imwrite("fgd.jpg", final_result1);

			con_img.copyTo(final_result1, pr_fgd_result1);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData1[row * 1920 + col] == 1){
						final_result1.at<Vec3b>(row, col)[0] = 255;
						final_result1.at<Vec3b>(row, col)[1] = 255;
						final_result1.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}

			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "1.jpg";
			imwrite(resultFileName.c_str(), final_result1);
			printf("%d번째 프레임, 1번째 BodyIndex GraphCut 완료\n", i);
		}

		
		if (isBody2){
			printf("%d번째 프레임, 2번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc2;

			gc2.graphCut(con_img, //입력영상		
				GC_Mask2,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap2,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap2);

			Mat fgd_result2, pr_fgd_result2;

			Mat foreground2(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground2(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result2(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask2, cv::GC_FGD, fgd_result2, cv::CMP_EQ);
			con_img.copyTo(foreground2, fgd_result2);

			compare(GC_Mask2, GC_PR_FGD, pr_fgd_result2, CMP_EQ);
			con_img.copyTo(pr_foreground2, pr_fgd_result2);

			con_img.copyTo(final_result2, fgd_result2);

			//imwrite("fgd.jpg", final_result);

			con_img.copyTo(final_result2, pr_fgd_result2);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData2[row * 1920 + col] == 1){
						final_result2.at<Vec3b>(row, col)[0] = 255;
						final_result2.at<Vec3b>(row, col)[1] = 255;
						final_result2.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}

			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "2.jpg";
			imwrite(resultFileName.c_str(), final_result2);

			printf("%d번째 프레임, 2번째 BodyIndex GraphCut 완료\n", i);
		}

		if (isBody3){
			printf("%d번째 프레임, 3번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc3;

			gc3.graphCut(con_img, //입력영상		
				GC_Mask3,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap3,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap3);

			Mat fgd_result3, pr_fgd_result3;

			Mat foreground3(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground3(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result3(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask3, cv::GC_FGD, fgd_result3, cv::CMP_EQ);
			con_img.copyTo(foreground3, fgd_result3);

			compare(GC_Mask3, GC_PR_FGD, pr_fgd_result3, CMP_EQ);
			con_img.copyTo(pr_foreground3, pr_fgd_result3);

			con_img.copyTo(final_result3, fgd_result3);

			con_img.copyTo(final_result3, pr_fgd_result3);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData3[row * 1920 + col] == 1){
						final_result3.at<Vec3b>(row, col)[0] = 255;
						final_result3.at<Vec3b>(row, col)[1] = 255;
						final_result3.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}

			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "3.jpg";
			imwrite(resultFileName.c_str(), final_result3);

			printf("%d번째 프레임, 3번째 BodyIndex GraphCut 완료\n", i);

		}

		if (isBody4){
			printf("%d번째 프레임, 4번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc4;

			gc4.graphCut(con_img, //입력영상		
				GC_Mask4,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap4,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap4);

			Mat fgd_result4, pr_fgd_result4;

			Mat foreground4(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground4(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result4(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask4, cv::GC_FGD, fgd_result4, cv::CMP_EQ);
			con_img.copyTo(foreground4, fgd_result4);

			compare(GC_Mask4, GC_PR_FGD, pr_fgd_result4, CMP_EQ);
			con_img.copyTo(pr_foreground4, pr_fgd_result4);

			con_img.copyTo(final_result4, fgd_result4);

			con_img.copyTo(final_result4, pr_fgd_result4);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData4[row * 1920 + col] == 1){
						final_result4.at<Vec3b>(row, col)[0] = 255;
						final_result4.at<Vec3b>(row, col)[1] = 255;
						final_result4.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}

			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "4.jpg";
			imwrite(resultFileName.c_str(), final_result4);

			printf("%d번째 프레임, 4번째 BodyIndex GraphCut 완료\n", i);
		}

		if (isBody5){
			printf("%d번째 프레임, 5번째 BodyIndex GraphCut 시작\n", i);
			my::GraphCut gc5;

			gc5.graphCut(con_img, //입력영상		
				GC_Mask5,//분할 마스크
				rect, //전경을 포함하는 직사각형
				skeletonMap5,
				back, fore, //모델
				1,//반복횟수
				GC_INIT_WITH_MASK,	//직사각형 사용
				skeletonMap5);

			Mat fgd_result5, pr_fgd_result5;

			Mat foreground5(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat pr_foreground5(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			Mat final_result5(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


			compare(GC_Mask5, cv::GC_FGD, fgd_result5, cv::CMP_EQ);
			con_img.copyTo(foreground5, fgd_result5);

			compare(GC_Mask0, GC_PR_FGD, pr_fgd_result5, CMP_EQ);
			con_img.copyTo(pr_foreground5, pr_fgd_result5);

			con_img.copyTo(final_result5, fgd_result5);

			con_img.copyTo(final_result5, pr_fgd_result5);

			for (int row = 0; row < 1080; row++){
				for (int col = 0; col < 1920; col++){
					if (skeletonData5[row * 1920 + col] == 1){
						final_result5.at<Vec3b>(row, col)[0] = 255;
						final_result5.at<Vec3b>(row, col)[1] = 255;
						final_result5.at<Vec3b>(row, col)[2] = 0;

					}
				}
			}

			std::string resultFileName = filePath::getInstance()->getResultPath() + "result" + to_string(i) + "5.jpg";
			imwrite(resultFileName.c_str(), final_result5);

			printf("%d번째 프레임, 5번째 BodyIndex GraphCut 완료\n", i);
		}
	}
	

	



	/*
	Mat test = imread("KinectScreenshot_RGB77.bmp");
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			if (skeletonData[row * 1920 + col] == 1){
				test.at<Vec3b>(row, col)[0] = 255;
				test.at<Vec3b>(row, col)[1] = 255;
				test.at<Vec3b>(row, col)[2] = 255;
			}
		}
	}
	imwrite("dafadf.jpg", test);
	*/
	
	/*
	my::GraphCut gc;

	gc.graphCut(con_img, //입력영상		
		GC_Mask,//분할 마스크
		rect, //전경을 포함하는 직사각형
		skeletonMap,
		back, fore, //모델
		1,//반복횟수
		GC_INIT_WITH_MASK,	//직사각형 사용
		skeletonMap);

	/*
	grabCut(con_img, //입력영상
		GC_Mask,//분할 마스크
		rect, //전경을 포함하는 직사각형
		back, fore, //모델
		1,//반복횟수
		GC_INIT_WITH_MASK);//직사각형 사용

*/

	
	/*
	Mat fgd_result, pr_fgd_result;

	Mat foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	Mat pr_foreground(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 255, 255));


	compare(GC_Mask, cv::GC_FGD, fgd_result, cv::CMP_EQ);	
	con_img.copyTo(foreground, fgd_result);
	
	compare(GC_Mask, GC_PR_FGD, pr_fgd_result, CMP_EQ);
	con_img.copyTo(pr_foreground, pr_fgd_result);

	con_img.copyTo(final_result, fgd_result);

	imwrite("fgd.jpg", final_result);

	con_img.copyTo(final_result, pr_fgd_result);

	namedWindow("Result");
	imshow("Result", final_result);

	//namedWindow("Foreground");
	//imshow("Foreground", foreground);
	
	imwrite("fgd+pr_fgd.jpg", final_result);
	waitKey(0);
	*/
	
	delete bodyIndexData;
	delete mappData;
	delete skeletonData0;
	delete skeletonData1;
	delete skeletonData2;
	delete skeletonData3;
	delete skeletonData4;
	delete skeletonData5;
	return 0;
}

void setSkeletonWeightMap(int frame){
	skeletonMap0 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	skeletonMap1 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	skeletonMap2 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	skeletonMap3 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	skeletonMap4 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	skeletonMap5 = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(0));
	
	double sigma = 20.0;
	const int range = 100;


	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){
			
			if (skeletonData0[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap0.at<uchar>(row, col + i))
							skeletonMap0.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap0.at<uchar>(row + i, col))
							skeletonMap0.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
			if (skeletonData1[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap1.at<uchar>(row, col + i))
							skeletonMap1.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap1.at<uchar>(row + i, col))
							skeletonMap1.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
			if (skeletonData2[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap2.at<uchar>(row, col + i))
							skeletonMap2.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap2.at<uchar>(row + i, col))
							skeletonMap2.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
			if (skeletonData3[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap3.at<uchar>(row, col + i))
							skeletonMap3.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap3.at<uchar>(row + i, col))
							skeletonMap3.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
			if (skeletonData4[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap4.at<uchar>(row, col + i))
							skeletonMap4.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap4.at<uchar>(row + i, col))
							skeletonMap4.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
			if (skeletonData5[row * IMAGE_WIDTH + col] == 1){
				for (int i = -1 * range; i <= range; i++){
					if (col + i >= 0 && col + i < IMAGE_WIDTH){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap5.at<uchar>(row, col + i))
							skeletonMap5.at<uchar>(row, col + i) = weight * 500;

					}
				}
				for (int i = -1 * range; i <= range; i++){
					if (+i >= 0 && row + i < IMAGE_HEIGHT){
						double weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));

						//printf("%f ", weight);
						if (weight >= skeletonMap5.at<uchar>(row + i, col))
							skeletonMap5.at<uchar>(row + i, col) = weight * 500;

					}
				}
			}
		}
	}

	/*
	double sigma = 20.0;
	const int range = 100;

	double* mask = new double[(2 * range) * (2 * range)];

	int index = 0;
	for (int i = -1 * range; i < range; i++){
		for (int j = -1 * range; j < range; j++)
		{
			//double weight = (1.0 / (sqrt(2 * PIE * sigma)) * exp((-1.0 * (powf(i, 2) + powf(j, 2)) / (2.0 * powf(sigma, 2))))) * 255;
			double x_weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(i, 2)) / (2.0 * powf(sigma, 2)))));
			double y_weight = (1.0 / (sqrt(2 * PIE) * sigma) * exp((-1.0 * (powf(j, 2)) / (2.0 * powf(sigma, 2)))));
			double weight = x_weight * y_weight * 10000;
			mask[index++] = weight;
			printf("%f ", weight);
		}
	}

	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){
			if (skeletonData[row * IMAGE_WIDTH + col] == 1){
				index = 0;
				for (int i = -1 * range; i <= range; i++){
					for (int j = -1 * range; j <= range; j++)
						if (col + i >= 0 && col + i < IMAGE_WIDTH &&
							row + j >= 0 && row + j < IMAGE_HEIGHT){
												
						//printf("%f ", weight);
						if (mask[index] >= skeletonMap.at<uchar>(row + j, col + i))
							skeletonMap.at<uchar>(row + j, col + i) = mask[index++];

						}
				}
			}
		}
	}
	*/


	std::string skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_0.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap0);

	skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_1.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap1);

	skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_2.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap2);

	skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_3.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap3);

	skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_4.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap4);

	skeletonFilePath = filePath::getInstance()->getResultPath() + "skeleton" + to_string(frame) + "_5.jpg";
	imwrite(skeletonFilePath.c_str(), skeletonMap5);
	
	printf("Weight Map 구함\n");
}

void loadDatas(int index){
	delete bodyIndexData;
	delete mappData;
	delete depthData;

	//BinaryReader br("FileHRbodyIndex_0.bin");
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

	//BinaryReader br2("FileMapp_1.bin");
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
/*


23 HAND_TIP_RIGHT																											21 HAND_TIP_LEFT
24 THUMB_RIGHT																												22 THUMB_LEFT
11 HAND_RIGHT											3 HEAD										7 HAND_LEFT

10 WRIST_RIGHT																		6 WRIST_LEFT

9 ELBOW_RIGHT						2 NECK						5 ELBOW_LEFT

8 SHOULDER_RIGHT		20 SPINE_SHOULDER		4 SHOULDER_LEFT

1 SPINE_MID

16 HIP_RIGHT    0 SPINE_BASE    12 HIP_LEFT

17 KNEE_RIGHT							13 KNEE_LEFT

18 ANKLE_RIGHT									14 ANKLE_LEFT

19 FOOT_RIGHT											15 FOOT_LEFT
*/


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
	delete skeletonData0;
	delete skeletonData1;
	delete skeletonData2;
	delete skeletonData3;
	delete skeletonData4;
	delete skeletonData5;
	
	skeletonData0 = new BYTE[1920 * 1080];
	skeletonData1 = new BYTE[1920 * 1080];
	skeletonData2 = new BYTE[1920 * 1080];
	skeletonData3 = new BYTE[1920 * 1080];
	skeletonData4 = new BYTE[1920 * 1080];
	skeletonData5 = new BYTE[1920 * 1080];

	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			skeletonData0[row * 1920 + col] = 0;
			skeletonData1[row * 1920 + col] = 0;
			skeletonData2[row * 1920 + col] = 0;
			skeletonData3[row * 1920 + col] = 0;
			skeletonData4[row * 1920 + col] = 0;
			skeletonData5[row * 1920 + col] = 0;
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
		
		if (bodyIndex == 9999 && index == 0) isBody0 = false;
		if (bodyIndex != 9999 && index == 0) isBody0 = true;

		if (bodyIndex == 9999 && index == 1) isBody1 = false;
		if (bodyIndex != 9999 && index == 1) isBody1 = true;

		if (bodyIndex == 9999 && index == 2) isBody2 = false;
		if (bodyIndex != 9999 && index == 2) isBody2 = true;

		if (bodyIndex == 9999 && index == 3) isBody3 = false;
		if (bodyIndex != 9999 && index == 3) isBody3 = true;

		if (bodyIndex == 9999 && index == 4) isBody4 = false;
		if (bodyIndex != 9999 && index == 4) isBody4 = true;

		if (bodyIndex == 9999 && index == 5) isBody5 = false;
		if (bodyIndex != 9999 && index == 5) isBody5 = true;
				
		
		skeletonPoints[count] = get(x, y);
		
		
		
		//if (atoi(getfield(tmp, 5)) == 9999){
			//printf("뛰어넘기\n");
			//continue;
		//}
		//printf("%d %d %d %d %d \n", x, y, z, state, bodyIndex);
		//printf("%d\n", x);
		
		//printf("Field 3 would be %s\n", getfield(tmp, 4));
		// NOTE strtok clobbers tmp
		


		count++;	
	}
	
	fclose(stream);
	/*
	skeletonPoints[0] = get(316, 190);
	skeletonPoints[1] = get(315, 135);
	skeletonPoints[2] = get(314, 80);
	skeletonPoints[3] = get(316, 53);
	skeletonPoints[4] = get(284, 101);
	skeletonPoints[5] = get(271, 143);
	skeletonPoints[6] = get(256, 186);
	skeletonPoints[7] = get(254, 201);
	skeletonPoints[8] = get(345, 102);
	skeletonPoints[9] = get(355, 147);
	skeletonPoints[10] = get(375, 185);
	skeletonPoints[11] = get(380, 201);
	skeletonPoints[12] = get(303, 191);
	skeletonPoints[13] = get(299, 260);
	skeletonPoints[14] = get(296, 323);
	skeletonPoints[15] = get(293, 340);
	skeletonPoints[16] = get(330, 190);
	skeletonPoints[17] = get(336, 255);
	skeletonPoints[18] = get(336, 322);
	skeletonPoints[19] = get(338, 340);
	skeletonPoints[20] = get(314, 94);
	skeletonPoints[21] = get(253, 214);
	skeletonPoints[22] = get(258, 201);
	skeletonPoints[23] = get(383, 213);
	skeletonPoints[24] = get(374, 208);


	//SPINE_BASE - SPINE_MID
	setSkeletonLine(skeletonPoints[0], skeletonPoints[1]);
	//SPINE_BASE - HIP_RIGHT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[16]);
	//SPINE_BASE - HIP_LEFT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[12]);
	//SPINE_MID - SPINE_SHOULDER
	setSkeletonLine(skeletonPoints[1], skeletonPoints[20]);
	//SPINE_SHOULDER - NECK
	setSkeletonLine(skeletonPoints[20], skeletonPoints[2]);
	//SPINE_SHOULDER - SHOULDER_RIGHT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[8]);
	//SHOULDER_RIGHT - SHOULDER_LEFT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[4]);
	//NECT - HEAD
	setSkeletonLine(skeletonPoints[2], skeletonPoints[3]);
	//SHOULDER_LEFT - ELBOW_RIGHT
	setSkeletonLine(skeletonPoints[8], skeletonPoints[9]);
	//ELBOW_RIGHT - WRIST_RIGHT
	setSkeletonLine(skeletonPoints[9], skeletonPoints[10]);
	//WRIST_RIGHT - HAND_RIGHT
	setSkeletonLine(skeletonPoints[10], skeletonPoints[11]);
	//HAND_RIGHT - HAND_TIP_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[23]);
	//HAND_RIGHT - THUMB_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[24]);
	//SHOULDER_LEFT - ELBOW_LEFT
	setSkeletonLine(skeletonPoints[4], skeletonPoints[5]);
	//ELBOW_LEFT - WRIST_LEFT
	setSkeletonLine(skeletonPoints[5], skeletonPoints[6]);
	//WRIST_LEFT - HAND_LEFT
	setSkeletonLine(skeletonPoints[6], skeletonPoints[7]);
	//HAND_LEFT - HAND_TIP_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[21]);
	//HAND_LEFT - THUMB_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[22]);
	//HIP_RIGHT - KNEE_RIGHT
	setSkeletonLine(skeletonPoints[16], skeletonPoints[17]);
	//KNEE_RIGHT - ANKLE_RIGHT
	setSkeletonLine(skeletonPoints[17], skeletonPoints[18]);
	//ANKLE_RIGHT - FOOT_RIGHT
	setSkeletonLine(skeletonPoints[18], skeletonPoints[19]);
	//HIP_LEFT - KNEE_LEFT
	setSkeletonLine(skeletonPoints[12], skeletonPoints[13]);
	//KNEE_LEFT - ANKLE_LEFT
	setSkeletonLine(skeletonPoints[13], skeletonPoints[14]);
	//ANKLE_LEFT - FOOT_LEFT
	setSkeletonLine(skeletonPoints[14], skeletonPoints[15]);
	/*
	skeletonPoints[0] = get(142, 216);
	skeletonPoints[1] = get(142, 161);
	skeletonPoints[2] = get(142, 109);
	skeletonPoints[3] = get(142, 84);
	skeletonPoints[4] = get(114, 130);
	skeletonPoints[5] = get(95, 162);
	skeletonPoints[6] = get(76, 186);
	skeletonPoints[7] = get(64, 197);
	skeletonPoints[8] = get(172, 135);
	skeletonPoints[9] = get(180, 168);
	skeletonPoints[10] = get(182, 191);
	skeletonPoints[11] = get(181, 203);
	skeletonPoints[12] = get(129, 216);
	skeletonPoints[13] = get(125, 265);
	skeletonPoints[14] = get(125, 322);
	skeletonPoints[15] = get(120, 338);
	skeletonPoints[16] = get(156, 215);
	skeletonPoints[17] = get(161, 268);
	skeletonPoints[18] = get(162, 319);
	skeletonPoints[19] = get(164, 336);
	skeletonPoints[20] = get(142, 121);
	skeletonPoints[21] = get(58, 210);
	skeletonPoints[22] = get(66, 192);
	skeletonPoints[23] = get(184, 211);
	skeletonPoints[24] = get(176, 204);

	//SPINE_BASE - SPINE_MID
	setSkeletonLine(skeletonPoints[0], skeletonPoints[1]);
	//SPINE_BASE - HIP_RIGHT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[16]);
	//SPINE_BASE - HIP_LEFT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[12]);
	//SPINE_MID - SPINE_SHOULDER
	setSkeletonLine(skeletonPoints[1], skeletonPoints[20]);
	//SPINE_SHOULDER - NECK
	setSkeletonLine(skeletonPoints[20], skeletonPoints[2]);
	//SPINE_SHOULDER - SHOULDER_RIGHT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[8]);
	//SHOULDER_RIGHT - SHOULDER_LEFT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[4]);
	//NECT - HEAD
	setSkeletonLine(skeletonPoints[2], skeletonPoints[3]);
	//SHOULDER_LEFT - ELBOW_RIGHT
	setSkeletonLine(skeletonPoints[8], skeletonPoints[9]);
	//ELBOW_RIGHT - WRIST_RIGHT
	setSkeletonLine(skeletonPoints[9], skeletonPoints[10]);
	//WRIST_RIGHT - HAND_RIGHT
	setSkeletonLine(skeletonPoints[10], skeletonPoints[11]);
	//HAND_RIGHT - HAND_TIP_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[23]);
	//HAND_RIGHT - THUMB_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[24]);
	//SHOULDER_LEFT - ELBOW_LEFT
	setSkeletonLine(skeletonPoints[4], skeletonPoints[5]);
	//ELBOW_LEFT - WRIST_LEFT
	setSkeletonLine(skeletonPoints[5], skeletonPoints[6]);
	//WRIST_LEFT - HAND_LEFT
	setSkeletonLine(skeletonPoints[6], skeletonPoints[7]);
	//HAND_LEFT - HAND_TIP_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[21]);
	//HAND_LEFT - THUMB_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[22]);
	//HIP_RIGHT - KNEE_RIGHT
	setSkeletonLine(skeletonPoints[16], skeletonPoints[17]);
	//KNEE_RIGHT - ANKLE_RIGHT
	setSkeletonLine(skeletonPoints[17], skeletonPoints[18]);
	//ANKLE_RIGHT - FOOT_RIGHT
	setSkeletonLine(skeletonPoints[18], skeletonPoints[19]);
	//HIP_LEFT - KNEE_LEFT
	setSkeletonLine(skeletonPoints[12], skeletonPoints[13]);
	//KNEE_LEFT - ANKLE_LEFT
	setSkeletonLine(skeletonPoints[13], skeletonPoints[14]);
	//ANKLE_LEFT - FOOT_LEFT
	setSkeletonLine(skeletonPoints[14], skeletonPoints[15]);
	*/
	/*
	skeletonPoints[0] = get(170, 211);
	skeletonPoints[1] = get(167, 158);
	skeletonPoints[2] = get(165, 104);
	skeletonPoints[3] = get(165, 83);
	skeletonPoints[4] = get(140, 129);
	skeletonPoints[5] = get(123, 175);
	skeletonPoints[6] = get(122, 211);
	skeletonPoints[7] = get(122, 223);
	skeletonPoints[8] = get(190, 130);
	skeletonPoints[9] = get(204, 175);
	skeletonPoints[10] = get(204, 208);
	skeletonPoints[11] = get(202, 218);
	skeletonPoints[12] = get(157, 210);
	skeletonPoints[13] = get(159, 270);
	skeletonPoints[14] = get(164, 336);
	skeletonPoints[15] = get(163, 344);
	skeletonPoints[16] = get(183, 212);
	skeletonPoints[17] = get(174, 269);
	skeletonPoints[18] = get(183, 311);
	skeletonPoints[19] = get(184, 322);
	skeletonPoints[20] = get(165, 117);
	skeletonPoints[21] = get(125, 234);
	skeletonPoints[22] = get(119, 230);
	skeletonPoints[23] = get(201, 229);
	skeletonPoints[24] = get(209, 218);

	//SPINE_BASE - SPINE_MID
	setSkeletonLine(skeletonPoints[0], skeletonPoints[1]);
	//SPINE_BASE - HIP_RIGHT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[16]);
	//SPINE_BASE - HIP_LEFT
	setSkeletonLine(skeletonPoints[0], skeletonPoints[12]);
	//SPINE_MID - SPINE_SHOULDER
	setSkeletonLine(skeletonPoints[1], skeletonPoints[20]);
	//SPINE_SHOULDER - NECK
	setSkeletonLine(skeletonPoints[20], skeletonPoints[2]);
	//SPINE_SHOULDER - SHOULDER_RIGHT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[8]);
	//SHOULDER_RIGHT - SHOULDER_LEFT
	setSkeletonLine(skeletonPoints[20], skeletonPoints[4]);
	//NECT - HEAD
	setSkeletonLine(skeletonPoints[2], skeletonPoints[3]);
	//SHOULDER_LEFT - ELBOW_RIGHT
	setSkeletonLine(skeletonPoints[8], skeletonPoints[9]);
	//ELBOW_RIGHT - WRIST_RIGHT
	setSkeletonLine(skeletonPoints[9], skeletonPoints[10]);
	//WRIST_RIGHT - HAND_RIGHT
	setSkeletonLine(skeletonPoints[10], skeletonPoints[11]);
	//HAND_RIGHT - HAND_TIP_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[23]);
	//HAND_RIGHT - THUMB_RIGHT
	setSkeletonLine(skeletonPoints[11], skeletonPoints[24]);
	//SHOULDER_LEFT - ELBOW_LEFT
	setSkeletonLine(skeletonPoints[4], skeletonPoints[5]);
	//ELBOW_LEFT - WRIST_LEFT
	setSkeletonLine(skeletonPoints[5], skeletonPoints[6]);
	//WRIST_LEFT - HAND_LEFT
	setSkeletonLine(skeletonPoints[6], skeletonPoints[7]);
	//HAND_LEFT - HAND_TIP_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[21]);
	//HAND_LEFT - THUMB_LEFT
	setSkeletonLine(skeletonPoints[7], skeletonPoints[22]);
	//HIP_RIGHT - KNEE_RIGHT
	setSkeletonLine(skeletonPoints[16], skeletonPoints[17]);
	//KNEE_RIGHT - ANKLE_RIGHT
	setSkeletonLine(skeletonPoints[17], skeletonPoints[18]);
	//ANKLE_RIGHT - FOOT_RIGHT
	setSkeletonLine(skeletonPoints[18], skeletonPoints[19]);
	//HIP_LEFT - KNEE_LEFT
	setSkeletonLine(skeletonPoints[12], skeletonPoints[13]);
	//KNEE_LEFT - ANKLE_LEFT
	setSkeletonLine(skeletonPoints[13], skeletonPoints[14]);
	//ANKLE_LEFT - FOOT_LEFT
	setSkeletonLine(skeletonPoints[14], skeletonPoints[15]);
	*/
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

	//printf("%d, %d = > %d, %d\n", _x, _y,  x, y);	

	return pos;
}
void set(int x, int y, int index){

	for (int off_y = -3; off_y <= 3; off_y++){
		for (int off_x = -3; off_x <= 3; off_x++){
			if (off_y + y >= 0 && off_y + y < 1080){
				if (off_x + x >= 0 && off_x + x < 1920){
					if (index == 0)
						skeletonData0[(y + off_y) * 1920 + (x + off_x)] = 1;
					else if (index == 1)
						skeletonData1[(y + off_y) * 1920 + (x + off_x)] = 1;
					else if (index == 2)
						skeletonData2[(y + off_y) * 1920 + (x + off_x)] = 1;
					else if (index == 3)
						skeletonData3[(y + off_y) * 1920 + (x + off_x)] = 1;
					else if (index == 4)
						skeletonData4[(y + off_y) * 1920 + (x + off_x)] = 1;
					else if (index == 5)
						skeletonData5[(y + off_y) * 1920 + (x + off_x)] = 1;
					else
						printf("index 설정 잘못! \n");
				}
			}
		}
	}
}