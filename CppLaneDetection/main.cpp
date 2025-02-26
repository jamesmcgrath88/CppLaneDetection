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

typedef struct
{
	float angle;
	int x1;
	int y1;
	int x2;
	int y2;
} Line_t;

static void CreateROIMask(cv::Mat& mask, int width, int height)
{
	mask = cv::Mat::zeros(height, width, CV_8U);

	std::vector<cv::Point> fillContSingle;
	fillContSingle.push_back(cv::Point(400, 700));
	fillContSingle.push_back(cv::Point(800, 300));
	fillContSingle.push_back(cv::Point(1000, 300));
	fillContSingle.push_back(cv::Point(1700, 700));

	std::vector<std::vector<cv::Point> > fillContAll;
	fillContAll.push_back(fillContSingle);

	cv::fillPoly(mask, fillContAll, cv::Scalar(255));
}

static void DoLaneDetection(cv::Mat originalFrame, cv::Mat frameToProcess, cv::Mat& outFrame)
{
	cv::Mat grayFrame;
	cv::Mat blurredFrame;
	cv::Mat cannyFrame;
	cv::Mat maskedCannyFrame;
	static cv::Mat roiMask;
	static bool initDone = false;
	bool tuning = false;
	static int fno = 0;

	if (frameToProcess.empty())
	{
		std::cerr << "ERROR! blank frame grabbed" << std::endl;
		return;
	}
	
	cv::cvtColor(frameToProcess, grayFrame, cv::COLOR_RGB2GRAY);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\grayFrmae.jpg", grayFrame);

	cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(9, 9), 0, 0);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\blurredFrmae.jpg", blurredFrame);

	if (initDone == false)
	{
		CreateROIMask(roiMask, grayFrame.cols, grayFrame.rows);
		//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\roiMask.jpg", roiMask);
		initDone = true;
	}

	cv::Canny(blurredFrame, cannyFrame, 100.0f, 120.0f);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\canny.jpg", cannyFrame);
	cv::bitwise_and(cannyFrame, cannyFrame, maskedCannyFrame, roiMask );
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\masked_canny.jpg", maskedCannyFrame);
	std::vector<cv::Vec4i> lines;

	if (tuning)
	{
		for (double rho = 1.0f; rho < 1.1f; rho += 1.0f)
		{
			for (int theta = 1.0f; theta < 1.1f; theta += 1.0f)
			{
				for (int threshold = 10; threshold < 110; threshold += 10)
				{
					for (double minLineLength = 20.0f; minLineLength < 100.0f; minLineLength += 20.0f)
					{
						for (double maxLineGap = 10.0f; maxLineGap < 100.0f; maxLineGap += 20.0f)
						{
							std::vector<cv::Vec4i> linesTuning;
							cv::HoughLinesP(maskedCannyFrame, linesTuning, rho, (theta * (CV_PI / 180.0f)), threshold, minLineLength, maxLineGap);

							cv::Mat blankImage;
							blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
							for (size_t i = 0; i < linesTuning.size(); i++)
							{
								cv::Vec4i l = linesTuning[i];
								cv::line(blankImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3);

							}
							cv::Mat outFrameTuning;
							cv::addWeighted(originalFrame, 0.8, blankImage, 1, 0.0, outFrameTuning);
							//std::ostringstream outFileName;
							//outFileName << "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\tuning\\out_";
							//outFileName << "rho" << rho << "_";
							//outFileName << "theta" << theta << "_";
							//outFileName << "threshold" << threshold << "_";
							//outFileName << "mll" << minLineLength << "_";
							//outFileName << "mlg" << maxLineGap << "_";
							//outFileName << ".jpg";
							//cv::imwrite(outFileName.str(), outFrameTuning);
						}
					}
				}
			}
		}
	}
	cv::HoughLinesP(maskedCannyFrame, lines, 1.0, CV_PI / 180, 20, 20.0, 50.0);

	cv::Mat blankImage;
	blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
	std::vector<Line_t> positiveSlopeLines;
	std::vector<Line_t> negativeSlopeLines;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::Point p1 = cv::Point(l[0], l[1]);
		cv::Point p2 = cv::Point(l[2], l[3]);
		if (p2.x != p1.x)
		{
			double slope = (p2.y - p1.y) / (double)(p2.x - p1.x);
			float lineLength = pow(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2), .5);
			if ( (lineLength > 30) && (slope != 0.0) )
			{
				float tanTheta = tan((abs(p2.y - p1.y)) / (abs(p2.x - p1.x))); // tan(theta) value
				float angle = atan(tanTheta) * 180 / CV_PI;
				if (abs(angle) < 85 && abs(angle) > 20)
				{	// Going to keep this line
					Line_t line;
					line.x1 = p1.x;
					line.y1 = p1.y;
					line.x2 = p2.x;
					line.y2 = p2.y;
					line.angle = angle;
					if (slope < 0.0) // negative slope
					{
						negativeSlopeLines.push_back(line);
					}
					else // positive slope
					{
						positiveSlopeLines.push_back(line);
					}
					//cv::line(blankImage, p1, p2, cv::Scalar(0, 255, 0), 3);
				}
			}
		}
	}

	cv::addWeighted(originalFrame, 0.8, blankImage, 1, 0.0, outFrame);

	std::ostringstream outFileName;
	outFileName << "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\mc\\out_";
	outFileName << "t" << fno++ << "_";
	outFileName << ".jpg";
	cv::imwrite(outFileName.str(), maskedCannyFrame);
}

static void ExtractYellowAndWhite(cv::Mat frame, cv::Mat& outFrame)
{
	cv::Mat hlsFrame;
	cv::cvtColor(frame, hlsFrame, cv::COLOR_BGR2HLS);

	cv::Mat whiteFrame;
	cv::inRange(hlsFrame, cv::Scalar(0,200,0), cv::Scalar(255,255,255), whiteFrame);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_white_frame.jpg", whiteFrame);

	cv::Mat yellowFrame;
	cv::inRange(hlsFrame, cv::Scalar(60, 35, 140), cv::Scalar(180, 255, 255), yellowFrame);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_yellow_frame.jpg", yellowFrame);

	cv::Mat colorMaskFrame;
	cv::bitwise_or(whiteFrame, yellowFrame, colorMaskFrame);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_color_mask_frame.jpg", colorMaskFrame);

	cv::bitwise_and(frame, frame, outFrame, colorMaskFrame);
	//cv::imwrite("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\out_color_masked_frame.jpg", outFrame);
}

int main(void)
{
	std::cout << "CPP Lane Detection!" << std::endl;
	
	bool videoMode = false;

	if (videoMode == false)
	{
		cv::Mat frame = cv::imread("C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\SingleCarraigeway\\image19443.jpg");
		cv::Mat colorMaskedFrame;
		ExtractYellowAndWhite(frame, colorMaskedFrame);
		cv::Mat outFrame;
		DoLaneDetection(frame, colorMaskedFrame, outFrame);
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
		int fno = 0;
		while (capture.read(frame))
		{
			cv::Mat colorMaskedFrame;
			ExtractYellowAndWhite(frame, colorMaskedFrame);
			cv::Mat outFrame;
			DoLaneDetection(frame, colorMaskedFrame, outFrame);
			videoOut.WriteFrameToOutputVideo(outFrame);

			/*if (fno++ > 300)
			{
				break;
			}*/
		}

		videoOut.SaveOutputVideo();
	}

	return 0;
}