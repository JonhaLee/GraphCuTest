/*
#pragma once


typedef unsigned char BYTE;

struct Point
{
	int x;
	int y;
};


class Skeleton
{
public:
	Skeleton(); 
	static BYTE* skeletonData;
	static Skeleton* instance;

private:

	const static int skeletonRange = 3;

	static Skeleton* getInstance();
	static void setSkeletonData(const Point& skeletonPos, int skletonNumber);
	static void setSkeletonLine(Point start, Point end);
	static void extendSkeletonRange(int x, int y);
	
};

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

