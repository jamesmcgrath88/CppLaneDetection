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

static void DoLaneDetection(cv::Mat frame, cv::Mat& outFrame)
{
	cv::Mat grayFrame;
	cv::Mat cannyFrame;
	cv::Mat maskedCannyFrame;
	cv::Mat roiMask;
	static bool initDone = false;

	if (frame.empty())
	{
		std::cerr << "ERROR! blank frame grabbed" << std::endl;
		return;
	}
	
	cv::cvtColor(frame, grayFrame, cv::COLOR_RGB2GRAY);

	if (initDone == false)
	{
		CreateROIMask(roiMask, grayFrame.cols, grayFrame.rows);
		cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\roiMask.jpg", roiMask);
		initDone = true;
	}

	cv::Canny(grayFrame, cannyFrame, 100.0f, 120.0f);
	cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\canny.jpg", cannyFrame);
	cv::bitwise_and(cannyFrame, roiMask, maskedCannyFrame);
	cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\masked_canny.jpg", maskedCannyFrame);
	std::vector<cv::Vec4i> lines;

	for (double rho = 1.0f; rho < 10.0f; rho += 1.0f)
	{
		for (int theta = 1.0f; theta < 10.0f; theta += 1.0f)
		{
			for (int threshold = 10; threshold < 310; threshold += 20)
			{
				for (double minLineLength = 20.0f; minLineLength < 100.0f; minLineLength += 20.0f)
				{
					for (double maxLineGap = 10.0f; maxLineGap < 100.0f; maxLineGap += 20.0f)
					{
						std::vector<cv::Vec4i> linesTuning;
						cv::HoughLinesP(maskedCannyFrame, linesTuning, rho, (theta * (CV_PI / 180.0f)), threshold, minLineLength, maxLineGap);
						//cv::HoughLinesP(maskedCannyFrame, lines, 1.0, CV_PI / 180, 200, 20.0, 100.0);

						cv::Mat blankImage;
						blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
						for (size_t i = 0; i < linesTuning.size(); i++)
						{
							cv::Vec4i l = linesTuning[i];
							cv::line(blankImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3);

						}
						cv::Mat outFrameTuning;
						cv::addWeighted(frame, 0.8, blankImage, 1, 0.0, outFrameTuning);
						std::ostringstream outFileName;
						outFileName << "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\tuning\\out_";
						outFileName << "rho" << rho << "_";
						outFileName << "theta" << theta << "_";
						outFileName << "threshold" << threshold << "_";
						outFileName << "mll" << minLineLength << "_";
						outFileName << "mlg" << maxLineGap << "_";
						outFileName << ".jpg";
						cv::imwrite(outFileName.str(), outFrameTuning);
					}
				}
			}
		}
	}

	cv::HoughLinesP(maskedCannyFrame, lines, 1.0, CV_PI / 180, 200, 20.0, 100.0);

	cv::Mat blankImage;
	blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(blankImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3);

	}

	cv::addWeighted(frame, 0.8, blankImage, 1, 0.0, outFrame);

	std::ostringstream outFileName;
	outFileName << "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_";
	outFileName << "t" << 200 << "_";
	outFileName << ".jpg";
	cv::imwrite(outFileName.str(), outFrame);
}

int main(void)
{
	std::cout << "CPP Lane Detection!" << std::endl;
	
	bool videoMode = false;

	if (videoMode == false)
	{
		cv::Mat frame = cv::imread("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\SingleCarraigeway\\image15430.jpg");
		cv::Mat rgbFrame;
		//cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
		cv::Mat outFrame;
		DoLaneDetection(frame, outFrame);
		cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_frame.jpg", outFrame);
	}
	else
	{
		std::string inputVideo("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\Singlecarriageway.mp4");
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
		while (capture.read(frame))
		{
			cv::Mat outFrame;
			DoLaneDetection(frame, outFrame);
			videoOut.WriteFrameToOutputVideo(outFrame);
		}

		videoOut.SaveOutputVideo();
	}

	return 0;
}