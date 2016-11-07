#include "MySkeleton.h"
/*
Skeleton* Skeleton::instance = nullptr;

Skeleton* Skeleton::getInstance(){
	if (instance == nullptr)
		instance = new Skeleton();

	return instance;
}

void Skeleton::setSkeletonData(const Point& skeletonPos, int skletonNumber){
	skeletonData = new BYTE[1920 * 1080];
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			skeletonData[row * 1920 + col] = 0;
		}
	}

	Point* skeletonPoints = new Point[25];
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
*/
/*
void Skeleton::setSkeletonLine(Point start, Point end){
	/*
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
			extendSkeletonRange(x1, y);
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
			extendSkeletonRange(x, y1);
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
				extendSkeletonRange(x, y);
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
				extendSkeletonRange(x, y);
			}
		}
	}
	
}

void Skeleton::extendSkeletonRange(int x, int y){

	for (int off_y = -skeletonRange; off_y <= skeletonRange; off_y++){
		for (int off_x = -skeletonRange; off_x <= skeletonRange; off_x++){
			if (off_y + y >= 0 && off_y + y < 1080){
				if (off_x + x >= 0 && off_x + x < 1920){
					skeletonData[(y + off_y) * 1920 + (x + off_x)] = 1;
				}
			}
		}
	}
}
*/