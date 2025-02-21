#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include "OutputVideo.h""

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
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			std::cerr << "ERROR! blank frame grabbed" << std::endl;
			break;
		}

		cv::cvtColor(frame, grayFrame, cv::COLOR_RGB2GRAY);

		videoOut.WriteFrameToOutputVideo(grayFrame);
	}

	videoOut.SaveOutputVideo();

	return 0;
}