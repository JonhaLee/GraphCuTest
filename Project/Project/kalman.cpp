
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <tchar.h>

int _tmain(int argc, _TCHAR* argv[])
{
	// Initialize Kalman filter object, window, number generator, etc
	cvNamedWindow("Kalman", 1);
	CvRandState rng;
	cvRandInit(&rng, 0, 1, -1, CV_RAND_UNI);

	IplImage* img = cvCreateImage(cvSize(500, 500), 8, 3);
	CvKalman* kalman = cvCreateKalman(2, 1, 0);

	// State is phi, delta_phi - angle and angular velocity
	// Initialize with random guess
	CvMat* x_k = cvCreateMat(2, 1, CV_32FC1);
	cvRandSetRange(&rng, 0, 0.1, 0);
	rng.disttype = CV_RAND_NORMAL;
	cvRand(&rng, x_k);


	// Process noise
	CvMat* w_k = cvCreateMat(2, 1, CV_32FC1);

	// Measurements, only one parameter for angle
	CvMat* z_k = cvCreateMat(1, 1, CV_32FC1);
	cvZero(z_k);

	// Transition matrix F describes model parameters at and k and k+1
	const float F[] = { 1, 1, 0, 1 };
	memcpy(kalman->transition_matrix->data.fl, F, sizeof(F));

	// kalman parameters 초기화
	cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1));
	cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-5));
	cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(1e-1));
	cvSetIdentity(kalman->error_cov_post, cvRealScalar(1));

	// Choose random initial state
	cvRand(&rng, kalman->state_post);

	// Make colors
	CvScalar yellow = CV_RGB(255, 255, 0);
	CvScalar white = CV_RGB(255, 255, 255);
	CvScalar red = CV_RGB(255, 0, 0);

	while (1){
		// Prediect 단계
		const CvMat* y_k = cvKalmanPredict(kalman, 0);

		// Generate Measurement (z_k)
		cvRandSetRange(&rng, 0, sqrt(kalman->measurement_noise_cov->data.fl[0]), 0);
		cvRand(&rng, z_k);
		cvMatMulAdd(kalman->measurement_matrix, x_k, z_k, z_k);


		cvZero(img);
		//센서 등을 통해 관측 된 값을 표시하는 부분(노랑색)
		cvCircle(img,
			cvPoint(cvRound(img->width / 2 + img->width / 3 * cos(z_k->data.fl[0])),
			cvRound(img->height / 2 - img->width / 3 * sin(z_k->data.fl[0]))),
			4, yellow);
		//칼만 필터를 통해 보정된 값을 표시하는 부분(하얀색)
		cvCircle(img,
			cvPoint(cvRound(img->width / 2 + img->width / 3 * cos(y_k->data.fl[0])),
			cvRound(img->height / 2 - img->width / 3 * sin(y_k->data.fl[0]))),
			4, white, 2);
		// 불규칙하게 움직이는 실제 값(빨강색)
		cvCircle(img,
			cvPoint(cvRound(img->width / 2 + img->width / 3 * cos(x_k->data.fl[0])),
			cvRound(img->height / 2 - img->width / 3 * sin(x_k->data.fl[0]))),
			4, red);
		cvShowImage("Kalman", img);


		// Update 단계
		cvKalmanCorrect(kalman, z_k);

		
		// x_k+1 = x_k * F + w_k 		
		cvRandSetRange(&rng, 0, sqrt(kalman->process_noise_cov->data.fl[0]), 0);
		cvRand(&rng, w_k);
		cvMatMulAdd(kalman->transition_matrix, x_k, w_k, x_k);

		// Exit on esc key
		if (cvWaitKey(100) == 27)
			break;
	}

	return 0;
}