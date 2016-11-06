#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "BinaryStream.h"

#include "gc.h"

#include <string.h>

using namespace cv;

BYTE* bodyIndexData;
BYTE* depthData;
short* mappData;
BYTE* skeletonData;

char* filepath = "C:\\Users\\Jonha\\Desktop\\Data8\\";

enum 
{
	none = 0,
	far = 0,
	middle_fal = 2,
	middle =  5,
	middle_near = 7,
	near = 9,
	origin = 10,
};


void ida();
void setSkeletonData();
void setSkeletonLine(Point2i start, Point2i end);
Point2i get(int _x, int _y);
void set(int x, int y);


int main(){
	printf("OpenCV Version : %s\n\n", CV_VERSION);

	//Mat con_img = imread("KinectScreenshot_RGB0.bmp");
	Mat con_img = imread("KinectScreenshot_RGB77.bmp");
	Mat result; // 분할 (4자기 가능한 값)
	//GrabCut에 사용되는 변수들
	Mat back, fore;	//모델(초기 사용)
	Rect rect(10, 10, 100, 100);

	ida();
	setSkeletonData();

	Mat GC_Mask(con_img.rows, con_img.cols, CV_8UC1, Scalar(GC_PR_BGD));
	
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 3840; col+=2){
			if (bodyIndexData[row * 3840 + col] < 6){
				if (bodyIndexData[row * 3840 + col + 1] == 1){
					GC_Mask.at<uchar>(row, (col / 2)) = GC_FGD;
				}
				else{
					//GC_Mask.at<uchar>(row, (col / 2)) = GC_PR_FGD;
					GC_Mask.at<uchar>(row, (col / 2)) = GC_PR_FGD;
				}
			}
		}
	}

	for (int row = 0; row < 1080; row++){
		for(int col = 0; col < 1920; col++){
			if (skeletonData[row * 1920 + col] == 1){
				GC_Mask.at<uchar>(row, col) = GC_FGD;
			}
		}
	}

	for (int row = 200; row < 875; row++){
		for (int i = 0; i < 10; i++){
			GC_Mask.at<uchar>(row, 1050 + i) = GC_BGD;
			GC_Mask.at<uchar>(row, 370 + i) = GC_BGD;
		}
	}

	Mat skeletonMap(con_img.rows, con_img.cols, CV_8UC1, Scalar(0));
	
	//스켈레톤 맵 만들기
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			if (skeletonData[row * 1920 + col] == 1){
				skeletonMap.at<uchar>(row, col) = origin;
				if (row - 1 >= 0) skeletonMap.at<uchar>(row - 1, col) > origin ? skeletonMap.at<uchar>(row - 1, col) : skeletonMap.at<uchar>(row - 1, col) = origin;
				if (row - 2 >= 0) skeletonMap.at<uchar>(row - 2, col) > near ? skeletonMap.at<uchar>(row - 2, col) : skeletonMap.at<uchar>(row - 2, col) = near;
				if (row - 3 >= 0) skeletonMap.at<uchar>(row - 3, col) > middle_near ? skeletonMap.at<uchar>(row - 3, col) : skeletonMap.at<uchar>(row - 3, col) = middle_near;
				if (row - 4 >= 0) skeletonMap.at<uchar>(row - 4, col) > middle ? skeletonMap.at<uchar>(row - 4, col) : skeletonMap.at<uchar>(row - 4, col) = middle;
				if (row - 5 >= 0) skeletonMap.at<uchar>(row - 5, col) > middle_fal ? skeletonMap.at<uchar>(row - 5, col) : skeletonMap.at<uchar>(row - 5, col) = middle_fal;
				if (row - 6 >= 0) skeletonMap.at<uchar>(row - 6, col) > far ? skeletonMap.at<uchar>(row - 6, col) : skeletonMap.at<uchar>(row - 6, col) = far;

				if (row + 1 >= 0) skeletonMap.at<uchar>(row + 1, col) > origin ? skeletonMap.at<uchar>(row + 1, col) : skeletonMap.at<uchar>(row + 1, col) = origin;
				if (row + 2 >= 0) skeletonMap.at<uchar>(row + 2, col) > near ? skeletonMap.at<uchar>(row + 2, col) : skeletonMap.at<uchar>(row + 2, col) = near;
				if (row + 3 >= 0) skeletonMap.at<uchar>(row + 3, col) > middle_near ? skeletonMap.at<uchar>(row + 3, col) : skeletonMap.at<uchar>(row + 3, col) = middle_near;
				if (row + 4 >= 0) skeletonMap.at<uchar>(row + 4, col) > middle ? skeletonMap.at<uchar>(row + 4, col) : skeletonMap.at<uchar>(row + 4, col) = middle;
				if (row + 5 >= 0) skeletonMap.at<uchar>(row + 5, col) > middle_fal ? skeletonMap.at<uchar>(row + 5, col) : skeletonMap.at<uchar>(row + 5, col) = middle_fal;
				if (row + 6 >= 0) skeletonMap.at<uchar>(row + 6, col) > far ? skeletonMap.at<uchar>(row + 6, col) : skeletonMap.at<uchar>(row + 6, col) = far;

				if (col - 1 >= 0) skeletonMap.at<uchar>(row, col - 1) > origin ? skeletonMap.at<uchar>(row, col - 1) : skeletonMap.at<uchar>(row, col - 1) = origin;
				if (col - 2 >= 0) skeletonMap.at<uchar>(row, col - 2) > near ? skeletonMap.at<uchar>(row, col - 2) : skeletonMap.at<uchar>(row, col - 2) = near;
				if (col - 3 >= 0) skeletonMap.at<uchar>(row, col - 3) > middle_near ? skeletonMap.at<uchar>(row, col - 3) : skeletonMap.at<uchar>(row, col - 3) = middle_near;
				if (col - 4 >= 0) skeletonMap.at<uchar>(row, col - 4) > middle ? skeletonMap.at<uchar>(row, col - 4) : skeletonMap.at<uchar>(row, col - 4) = middle;
				if (col - 5 >= 0) skeletonMap.at<uchar>(row, col - 5) > middle_fal ? skeletonMap.at<uchar>(row, col - 5) : skeletonMap.at<uchar>(row, col - 5) = middle_fal;
				if (col - 6 >= 0) skeletonMap.at<uchar>(row, col - 6) > far ? skeletonMap.at<uchar>(row, col - 6) : skeletonMap.at<uchar>(row, col - 6) = far;

				if (col + 1 >= 0) skeletonMap.at<uchar>(row, col + 1) > origin ? skeletonMap.at<uchar>(row, col + 1) : skeletonMap.at<uchar>(row, col + 1) = origin;
				if (col + 2 >= 0) skeletonMap.at<uchar>(row, col + 2) > near ? skeletonMap.at<uchar>(row, col + 2) : skeletonMap.at<uchar>(row, col + 2) = near;
				if (col + 3 >= 0) skeletonMap.at<uchar>(row, col + 3) > middle_near ? skeletonMap.at<uchar>(row, col + 3) : skeletonMap.at<uchar>(row, col + 3) = middle_near;
				if (col + 4 >= 0) skeletonMap.at<uchar>(row, col + 4) > middle ? skeletonMap.at<uchar>(row, col + 4) : skeletonMap.at<uchar>(row, col + 4) = middle;
				if (col + 5 >= 0) skeletonMap.at<uchar>(row, col + 5) > middle_fal ? skeletonMap.at<uchar>(row, col + 5) : skeletonMap.at<uchar>(row, col + 5) = middle_fal;
				if (col + 6 >= 0) skeletonMap.at<uchar>(row, col + 6) > far ? skeletonMap.at<uchar>(row, col + 6) : skeletonMap.at<uchar>(row, col + 6) = far;
			}
		}
	}
	
	imwrite("skeletonMap.jpg", skeletonMap);

	Mat bodyIndex(con_img.rows, con_img.cols, CV_8UC1, Scalar(255));
	for (int row = 0; row < con_img.rows; row++){
		for (int col = 0; col < con_img.cols; col++){
			bodyIndex.at<uchar>(row, col) = bodyIndexData[row * 3840 + (col * 2) ];
		}
	}

	imwrite("bodyIndex.jpg", bodyIndex);

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
	Mat depth(424, 512, CV_8UC1, Scalar(0));

	for (int row = 0; row < 424; row++){
		for (int col = 0; col < 512; col++){
			depth.at<uchar>(row,col) = depthData[row * 512 + col];
			
		}
	}
	imwrite("124125.jpg", depth);
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
	delete skeletonData;
	return 0;
}

void ida(){
	
	//BinaryReader br("FileHRbodyIndex_0.bin");
	BinaryReader br("FileHRbodyIndex_77.bin");
	int pos = 0;
	int length = (int)3840 * 1080;

	bodyIndexData = new BYTE[3840 * 1080];
	
	int index = 0;
	while (pos < length)
	{
		bodyIndexData[index] = br.ReadBYTE();		

		index++;
		pos += sizeof(BYTE);
	}

	//BinaryReader br2("FileMapp_1.bin");
	BinaryReader br2("FileMapp_77.bin");
	pos = 0;
	length = (int)1024 * 424;

	mappData = new short[1024 * 424];

	index = 0;
	while (pos < length)
	{
		mappData[index] = br2.ReadInt16();
		index++;
		pos ++;
	}

	BinaryReader br3("Filedepth_77.bin");
	pos = 0;
	length = (int)512 * 424;

	depthData = new BYTE[512 * 424];

	index = 0;
	while (pos < length)
	{
		depthData[index] = br3.ReadBYTE();

		index++;
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
void setSkeletonData(){
	skeletonData = new BYTE[1920 * 1080];
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			skeletonData[row * 1920 + col] = 0;
		}
	}



	Point2i* skeletonPoints = new Point2i[25];
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
}

void setSkeletonLine(Point2i start, Point2i end){
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
			set(x1, y);
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
			set(x, y1);
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
				set(x, y);
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
				set(x, y);
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
void set(int x, int y){

	for (int off_y = -3; off_y <= 3; off_y++){
		for (int off_x = -3; off_x <= 3; off_x++){
			if (off_y + y >= 0 && off_y + y < 1080){
				if (off_x + x >= 0 && off_x + x < 1920){
					skeletonData[(y + off_y) * 1920 + (x + off_x)] = 1;
				}
			}
		}
	}
}