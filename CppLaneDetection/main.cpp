#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <string>
#include <iostream>
#include <vector>
#include "OutputVideo.h"

static void CreateROIMask(cv::Mat& mask, int width, int height)
{
	mask = cv::Mat::zeros(height, width, CV_8U);

	std::vector<cv::Point> fillContSingle;
	fillContSingle.push_back(cv::Point(400, 700));
	fillContSingle.push_back(cv::Point(900, 300));
	fillContSingle.push_back(cv::Point(920, 300));
	fillContSingle.push_back(cv::Point(1700, 700));

	std::vector<std::vector<cv::Point> > fillContAll;
	fillContAll.push_back(fillContSingle);

	cv::fillPoly(mask, fillContAll, cv::Scalar(255));
}

int main(void)
{
	std::cout << "Hello World!" << std::endl;

	//cv::Mat stormySea = cv::imread("C:\\dev\\CppLaneDetection\\CppLaneDetection\\stormy_sea.jpg");
	//cv::imshow("Stormy Sea", stormySea);
	//cv::waitKey(0);

	std::string inputVideo("C:\\Users\\james\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\Singlecarriageway.mp4");
	cv::VideoCapture capture(inputVideo);
	if (!capture.isOpened())
	{
		std::cout << "Failed to open " << inputVideo << std::endl;
		return 1;
	}
	else
	{
		std::cout << "Successfully opened " << inputVideo << std::endl;
	}

	OutputVideo videoOut;

	cv::Mat frame;
	cv::Mat grayFrame;
	cv::Mat cannyFrame;
	cv::Mat maskedCannyFrame;
	cv::Mat roiMask;
	bool initDone = false;
	int frameCount = 0;

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			std::cerr << "ERROR! blank frame grabbed" << std::endl;
			break;
		}

		cv::cvtColor(frame, grayFrame, cv::COLOR_RGB2GRAY);

		if (initDone == false)
		{
			CreateROIMask(roiMask, grayFrame.cols, grayFrame.rows);
			initDone = true;
		}

		cv::Canny(grayFrame, cannyFrame, 100.0f, 200.0f);

		cv::bitwise_and( cannyFrame, roiMask, maskedCannyFrame );

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(maskedCannyFrame, lines, 2.0, CV_PI / 180, 500, 0.0, 100.0);

		cv::Mat blankImage;
		blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
		for (size_t i = 0; i < lines.size(); i++)
		{
			cv::Vec4i l = lines[i];
			cv::line(blankImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3);

		}

		cv::Mat outFrame;
		cv::addWeighted(frame, 0.8, blankImage, 1, 0.0, outFrame);

		//videoOut.WriteFrameToOutputVideo(roiMask);
		videoOut.WriteFrameToOutputVideo(outFrame);

		//if (++frameCount > 300) { break; }
	}

	videoOut.SaveOutputVideo();

	return 0;
}