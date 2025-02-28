#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include "OutputVideo.h"

const std::string WORKING_DIRECTORY = "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\";
const bool SAVE_EVERYTHING = false;
const bool PRINT_TO_CONSOLE = false;

typedef struct
{
	float angle;
	cv::Point p1;
	cv::Point p2;
	int bbP1Y; // Lane bounding box coordinates - p1.y
	double bbXIntercept; // Lane bounding box coordinates - x intercept
	double bbYIntercept; // Lane bounding box coordinates - y intercept
	float bbSlope;
} Line_t;

typedef struct
{
	double rho;
	double theta;
	int threshold;
	double minLineLength;
	double maxLineGap;
} HoughHyperparameters_t;

typedef struct
{
	double t1;
	double t2;
} CannyHyperparameters_t;

typedef struct
{
	HoughHyperparameters_t houghHyperparams;
	CannyHyperparameters_t cannyHyperparams;
} LaneDetectionHyperparameters;

bool compareBySlope(const Line_t& a, const Line_t& b)
{
	return a.angle < b.angle;
}
bool compareByXIntercept(const Line_t& a, const Line_t& b)
{
	return a.bbXIntercept < b.bbXIntercept;
}

cv::Point FindIntersection(cv::Point A, cv::Point B, cv::Point C, cv::Point D)
{
	// Line AB represented as a1x + b1y = c1
	double a = B.y - A.y;
	double b = A.x - B.x;
	double c = a * (A.x) + b * (A.y);
	// Line CD represented as a2x + b2y = c2
	double a1 = D.y - C.y;
	double b1 = C.x - D.x;
	double c1 = a1 * (C.x) + b1 * (C.y);
	double det = a * b1 - a1 * b;
	if (det == 0)
	{
		return cv::Point(0, 0);
	}
	else
	{
		int x = (int)std::round( (b1 * c - b * c1) / det );
		int y = (int)std::round( (a * c1 - a1 * c) / det );
		return cv::Point(x, y);
	}
}

static int CreateROIMask(cv::Mat& mask, int width, int height)
{
	int roiMaskBBLowerY = 700;

	mask = cv::Mat::zeros(height, width, CV_8U);

	std::vector<cv::Point> fillContSingle;
	fillContSingle.push_back(cv::Point(400, roiMaskBBLowerY));
	fillContSingle.push_back(cv::Point(800, 300));
	fillContSingle.push_back(cv::Point(1000, 300));
	fillContSingle.push_back(cv::Point(1700, roiMaskBBLowerY));

	std::vector<std::vector<cv::Point> > fillContAll;
	fillContAll.push_back(fillContSingle);

	cv::fillPoly(mask, fillContAll, cv::Scalar(255));

	return roiMaskBBLowerY;
}

static void HoughTuning(cv::Mat originalFrame, cv::Mat maskedCannyFrame, bool saveEachFrame, int fID, HoughHyperparameters_t& bestParamsOut)
{
	for (double rho = 1.0; rho < 1.1; rho += 1.0)
	{
		for (int theta = 1; theta <= 1; theta++)
		{
			for (int threshold = 10; threshold < 110; threshold += 10)
			{
				for (double minLineLength = 20.0; minLineLength < 100.0; minLineLength += 20.0)
				{
					for (double maxLineGap = 10.0; maxLineGap < 100.0; maxLineGap += 20.0)
					{
						std::vector<cv::Vec4i> linesTuning;
						cv::HoughLinesP(maskedCannyFrame, linesTuning, rho, (theta * (CV_PI / 180.0f)), threshold, minLineLength, maxLineGap);

						cv::Mat blankImage;
						blankImage = cv::Mat::zeros(originalFrame.rows, originalFrame.cols, CV_8UC3);
						for (size_t i = 0; i < linesTuning.size(); i++)
						{
							cv::Vec4i l = linesTuning[i];
							cv::line(blankImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3);

						}
						cv::Mat outFrameTuning;
						cv::addWeighted(originalFrame, 0.8, blankImage, 1, 0.0, outFrameTuning);
						if (saveEachFrame)
						{
							std::ostringstream outFileName;
							outFileName << "C:\\Users\\jmcgrath\\Documents\\AutomotiveAI\\MVGCV\\Individual Assignment\\tuning\\out_";
							outFileName << "fID" << fID << "_";
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
	}
}

static void LineFiltering(std::vector<cv::Vec4i>& lines, int bbY, std::vector<Line_t>& positiveSlopeLines, std::vector<Line_t>& negativeSlopeLines)
{
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::Point p1 = cv::Point(l[0], l[1]);
		cv::Point p2 = cv::Point(l[2], l[3]);
		if (p2.x != p1.x)
		{
			double slope = (p2.y - p1.y) / (double)(p2.x - p1.x);
			double lineLength = pow(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2), .5);

			if ((lineLength > 30.0) && (slope != 0.0))
			{
				float tanTheta = tan((float)(abs(p2.y - p1.y)) / (float)(abs(p2.x - p1.x)));
				double angle = atan(tanTheta) * 180 / CV_PI;
				if (PRINT_TO_CONSOLE) { std::cout << "Line[" << i << "] - length: " << lineLength << " - slope: " << slope << " - angle: " << angle << std::endl; }
				if (abs(angle) < 85 && abs(angle) > 20)
				{	// Going to keep this line
					Line_t line;
					line.p1 = p1;
					line.p2 = p2;
					line.angle = (float)angle;
					line.bbP1Y = bbY - p1.y;
					int bbP2Y = bbY - p2.y;
					line.bbSlope = (float)(bbP2Y - line.bbP1Y)/(p2.x - p1.x);
					line.bbYIntercept = line.bbP1Y - line.bbSlope * p1.x;
					line.bbXIntercept = (-1.0f * line.bbYIntercept) / line.bbSlope;
					if (line.bbSlope < 0.0f) // negative slope
					{
						negativeSlopeLines.push_back(line);
					}
					else // positive slope
					{
						positiveSlopeLines.push_back(line);
					}
				}
			}
		}
	}
}

static float SlopeFiltering(std::vector<Line_t>& lines)
{
	float slopeMean = 0.0f;

	if (lines.size() > 0)
	{
		float slopeMedian = 0.0f;
		std::sort(lines.begin(), lines.end(), compareBySlope);
		size_t middleIdx = lines.size() / 2;
		if (lines.size() % 2 == 0) // Even number of elements in the vector
		{
			slopeMedian = (lines[middleIdx].angle + lines[(middleIdx - 1)].angle) / 2.0f;
		}
		else
		{
			slopeMedian = lines[middleIdx].angle;
		}
		if (PRINT_TO_CONSOLE)
		{
			std::cout << "Number of slope lines: " << lines.size() << std::endl;
			std::cout << "Slope angle median: " << slopeMedian << std::endl;
		}
		float slopeSum = 0.0f;
		float slopeFilter = (slopeMedian * 0.2f);
		std::vector<Line_t>::iterator it = lines.begin();
		while (it != lines.end())
		{
			float dA = abs(it->angle - slopeMedian);
			if (dA > slopeFilter)
			{
				it = lines.erase(it);
			}
			else
			{
				slopeSum += it->bbSlope;
				++it;
			}
		}

		if (lines.size() > 0)
		{
			slopeMean = slopeSum / lines.size();
			if (PRINT_TO_CONSOLE) { std::cout << "Mean slope: " << slopeMean << std::endl; }
		}
	}

	return slopeMean;
}

static void xInterceptFiltering(std::vector<Line_t>& lines, float& meanXInterceptOut, int& yOut)
{
	std::sort(lines.begin(), lines.end(), compareByXIntercept);
	double xInterceptMedian = 0.0f;
	size_t middleIdx = lines.size() / 2;
	if (lines.size() % 2 == 0) // Even number of elements in the vector
	{
		xInterceptMedian = (lines[middleIdx].bbXIntercept + lines[(middleIdx - 1)].bbXIntercept) / 2.0f;
	}
	else
	{
		xInterceptMedian = lines[middleIdx].bbXIntercept;
	}

	double xInterceptSum = 0.0f;
	int goodLinesCount = 0;
	for (const Line_t& i : lines)
	{
		if (abs(i.bbXIntercept - xInterceptMedian) < (0.35f * xInterceptMedian))
		{
			xInterceptSum += i.bbXIntercept;
			goodLinesCount++;
			if (i.p1.y < yOut)
			{
				yOut = i.p1.y;
			}
			else if (i.p2.y < yOut)
			{
				yOut = i.p2.y;
			}
		}
	}
	if (goodLinesCount > 0)
	{
		meanXInterceptOut = (float)(xInterceptSum / (float)goodLinesCount);
	}
}

static bool DoLaneDetection(cv::Mat originalFrame, cv::Mat frameToProcess, const cv::Mat& roiMask, int bbY, int fID, const LaneDetectionHyperparameters& hyperparams, cv::Mat& outFrame, std::vector<cv::Point>& boudingBoxVertices)
{
	cv::Mat grayFrame;
	cv::Mat blurredFrame;
	cv::Mat cannyFrame;
	cv::Mat maskedCannyFrame;
	bool laneFound = false;

	if (frameToProcess.empty())
	{
		std::cerr << "ERROR! blank frame grabbed" << std::endl;
		return laneFound;
	}
	
	cv::cvtColor(frameToProcess, grayFrame, cv::COLOR_RGB2GRAY);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "grayFrame.jpg", grayFrame);
	}

	cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(9, 9), 0, 0);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "blurredFrame.jpg", blurredFrame);
	}

	cv::Canny(blurredFrame, cannyFrame, hyperparams.cannyHyperparams.t1, hyperparams.cannyHyperparams.t2);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "canny.jpg", cannyFrame);
	}
	cv::bitwise_and(cannyFrame, cannyFrame, maskedCannyFrame, roiMask );
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "masked_canny.jpg", maskedCannyFrame);
	}

	std::vector<cv::Vec4i> lines;

	cv::HoughLinesP(maskedCannyFrame, lines, hyperparams.houghHyperparams.rho,
		(hyperparams.houghHyperparams.theta*(CV_PI / 180.0f)), hyperparams.houghHyperparams.threshold,
		hyperparams.houghHyperparams.minLineLength, hyperparams.houghHyperparams.maxLineGap);

	cv::Mat blankImage;
	blankImage = cv::Mat::zeros(grayFrame.rows, grayFrame.cols, CV_8UC3);
	std::vector<Line_t> positiveSlopeLines;
	std::vector<Line_t> negativeSlopeLines;
	if (PRINT_TO_CONSOLE) { std::cout << "Found lines: " << lines.size() << std::endl; }
	// Line Filtering
	LineFiltering(lines, bbY, positiveSlopeLines, negativeSlopeLines);

	// Slope Filtering - multiply slopes by -1.0f to convert from bb coordinates back to image coordinates
	float posSlopeMean = -1.0f*SlopeFiltering(positiveSlopeLines);
	float negSlopeMean = -1.0f*SlopeFiltering(negativeSlopeLines);

	if ((positiveSlopeLines.size() > 0) && (negativeSlopeLines.size() > 0))
	{

		float meanXInterceptPos = 0.0f, meanXInterceptNeg = 0.0f;
		int yPos = originalFrame.rows, yNeg = originalFrame.rows;
		xInterceptFiltering(positiveSlopeLines, meanXInterceptPos, yPos);
		xInterceptFiltering(negativeSlopeLines, meanXInterceptNeg, yNeg);

		if ((yPos < originalFrame.rows) && (yNeg < originalFrame.rows))
		{
			cv::Point pos1, pos2;
			pos1.x = (int)std::round(meanXInterceptPos);
			pos1.y = bbY;
			pos2.y = yPos;
			pos2.x = (int)std::round(((float)(pos2.y - pos1.y) / posSlopeMean) + pos1.x);

			cv::Point neg1, neg2;
			neg1.x = (int)std::round(meanXInterceptNeg);
			neg1.y = bbY;
			neg2.y = yNeg;
			neg2.x = (int)std::round(((float)(neg2.y - neg1.y) / negSlopeMean) + neg1.x);

			if (pos2.x > neg2.x)
			{	// The lines cross - have them meet at the crossing point
				cv::Point intersection = FindIntersection( pos1, pos2, neg1, neg2 );
				pos2 = intersection;
				neg2 = intersection;
			}

			boudingBoxVertices.push_back(pos1);
			boudingBoxVertices.push_back(pos2);
			boudingBoxVertices.push_back(neg2);
			boudingBoxVertices.push_back(neg1);

			std::vector<std::vector<cv::Point> > fillContAll;
			fillContAll.push_back(boudingBoxVertices);
			cv::fillPoly(blankImage, fillContAll, cv::Scalar(0, 255, 255));
			cv::addWeighted(originalFrame, 0.8, blankImage, 0.2, 0.0, outFrame);

			cv::line(outFrame, pos1, pos2, cv::Scalar(0, 255, 0), 5);
			cv::line(outFrame, neg1, neg2, cv::Scalar(0, 255, 0), 5);

			laneFound = true;
		}
	}
	else
	{
		outFrame = originalFrame.clone();
	}

	return laneFound;
}

static void ExtractYellowAndWhite(cv::Mat frame, cv::Mat& outFrame)
{
	cv::Mat hlsFrame;
	cv::cvtColor(frame, hlsFrame, cv::COLOR_BGR2HLS);

	cv::Mat whiteFrame;
	cv::inRange(hlsFrame, cv::Scalar(0,200,0), cv::Scalar(255,255,255), whiteFrame);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "out_white_frame.jpg", whiteFrame);
	}

	cv::Mat yellowFrame;
	cv::inRange(hlsFrame, cv::Scalar(60, 35, 140), cv::Scalar(180, 255, 255), yellowFrame);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "out_yellow_frame.jpg", yellowFrame);
	}

	cv::Mat colorMaskFrame;
	cv::bitwise_or(whiteFrame, yellowFrame, colorMaskFrame);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "out_color_mask_frame.jpg", colorMaskFrame);
	}

	cv::bitwise_and(frame, frame, outFrame, colorMaskFrame);
	if (SAVE_EVERYTHING)
	{
		cv::imwrite(WORKING_DIRECTORY + "out_color_masked_frame.jpg", outFrame);
	}
}

int main(void)
{
	std::cout << "CPP Lane Detection!" << std::endl;
	
	bool videoMode = true;
	cv::Mat roiMask;
	int bbY = 0;
	std::vector<cv::Point> boudingBoxVertices;

	// Set some default hyperparameters
	LaneDetectionHyperparameters hyperparams;
	hyperparams.cannyHyperparams.t1 = 100.0f;
	hyperparams.cannyHyperparams.t2 = 120.0f;
	hyperparams.houghHyperparams.rho = 1.0f;
	hyperparams.houghHyperparams.theta = 1.0f;
	hyperparams.houghHyperparams.threshold = 20;
	hyperparams.houghHyperparams.minLineLength = 20.0f;
	hyperparams.houghHyperparams.maxLineGap = 50.0f;

	if (videoMode == false)
	{
		int fID = 19131;//19131//29741
		cv::Mat frame = cv::imread(WORKING_DIRECTORY + "SingleCarraigeway\\image" + std::to_string(fID) + ".jpg");
		bbY = CreateROIMask(roiMask, frame.cols, frame.rows);
		if (SAVE_EVERYTHING)
		{
			cv::imwrite(WORKING_DIRECTORY + "roiMask.jpg", roiMask);
		}
		cv::Mat colorMaskedFrame;
		ExtractYellowAndWhite(frame, colorMaskedFrame);
		cv::Mat outFrame;
		DoLaneDetection(frame, colorMaskedFrame, roiMask, bbY, fID, hyperparams, outFrame, boudingBoxVertices);
		cv::imwrite(WORKING_DIRECTORY + "out_frame_" + std::to_string(fID) + ".jpg", outFrame);
	}
	else
	{
		std::string inputVideo(WORKING_DIRECTORY + "Singlecarriageway.mp4");
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

		OutputVideo videoOut(WORKING_DIRECTORY);
		cv::Mat frame;
		int fno = 0;
		bool roiMaskInitDone = false;
		int frameLimit = 300;// 40000;
		while (capture.read(frame))
		{
			if (roiMaskInitDone == false)
			{
				bbY = CreateROIMask(roiMask, frame.cols, frame.rows);
				roiMaskInitDone = true;
			}
			cv::Mat colorMaskedFrame;
			ExtractYellowAndWhite(frame, colorMaskedFrame);
			cv::Mat outFrame;
			boudingBoxVertices.clear();
			DoLaneDetection(frame, colorMaskedFrame, roiMask, bbY, fno, hyperparams, outFrame, boudingBoxVertices);
			videoOut.WriteFrameToOutputVideo(outFrame);
			std::cout << "F#: " << fno << std::endl;
			fno++;
			if (fno > frameLimit)
			{	// Provide a way to break out of the processing is taking too long
				break;
			}
		}

		videoOut.SaveOutputVideo();
	}

	return 0;
}