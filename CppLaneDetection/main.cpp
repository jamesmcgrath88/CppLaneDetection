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

double GetTriangleAreaFromVertices(cv::Point v1, cv::Point v2, cv::Point v3)
{
	return (double)std::abs(v1.x * (v2.y - v3.y) + v2.x * (v3.y - v1.y) + v3.x * (v1.y - v2.y)) / 2.0;
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

static int CreateROIMask(cv::Mat& mask, int width, int height, std::vector<cv::Point>& roiVertices)
{
	int roiMaskBBLowerY = roiVertices[0].y;

	mask = cv::Mat::zeros(height, width, CV_8U);

	std::vector<std::vector<cv::Point> > fillContAll;
	fillContAll.push_back(roiVertices);

	cv::fillPoly(mask, fillContAll, cv::Scalar(255));

	return roiMaskBBLowerY;
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

static float CalculateIoU(int imCols, int imRows, std::vector<cv::Point> boudingBoxVertices, std::vector<cv::Point> groundTruthVertices)
{
	int unionCount = 0, intersectionCount = 0;
	if ((boudingBoxVertices.size() > 0) && (groundTruthVertices.size() > 0))
	{
		double boundingBoxArea = 0.0f;
		boundingBoxArea += GetTriangleAreaFromVertices(boudingBoxVertices[0], boudingBoxVertices[1], boudingBoxVertices[2]);
		boundingBoxArea += GetTriangleAreaFromVertices(boudingBoxVertices[0], boudingBoxVertices[2], boudingBoxVertices[3]);

		double groundTruthArea = 0.0f;
		groundTruthArea += GetTriangleAreaFromVertices(groundTruthVertices[0], groundTruthVertices[1], groundTruthVertices[2]);
		groundTruthArea += GetTriangleAreaFromVertices(groundTruthVertices[0], groundTruthVertices[2], groundTruthVertices[3]);


		for (int r = 0; r < imRows; r++)
		{
			for (int c = 0; c < imCols; c++)
			{
				double pBBarea = 0.0, pGTArea = 0.0;
				for (int i = 0; i < 4; i++)
				{
					pBBarea += GetTriangleAreaFromVertices(cv::Point(r, c), boudingBoxVertices[i], boudingBoxVertices[((i + 1) % 4)]);
					pGTArea += GetTriangleAreaFromVertices(cv::Point(r, c), groundTruthVertices[i], groundTruthVertices[((i + 1) % 4)]);
				}

				if ((pBBarea <= boundingBoxArea) && (pGTArea <= groundTruthArea))
				{
					intersectionCount++;
					unionCount++;
				}
				else if ((pBBarea <= boundingBoxArea) || (pGTArea <= groundTruthArea))
				{
					unionCount++;
				}
			}
		}
	}
	float iou = 0.0f;
	if (unionCount > 0)
	{
		iou = (float)intersectionCount / (float)unionCount;
	}
	return iou;
}

static float CalculateIoU(const cv::Mat& gtframe, std::vector<cv::Point> boudingBoxVertices)
{
	int unionCount = 0, intersectionCount = 0;

	if (boudingBoxVertices.size() > 0)
	{
		double boundingBoxArea = 0.0f;
		boundingBoxArea += GetTriangleAreaFromVertices(boudingBoxVertices[0], boudingBoxVertices[1], boudingBoxVertices[2]);
		boundingBoxArea += GetTriangleAreaFromVertices(boudingBoxVertices[0], boudingBoxVertices[2], boudingBoxVertices[3]);

		for (int r = 0; r < gtframe.rows; r++)
		{
			for (int c = 0; c < gtframe.cols; c++)
			{
				bool pointInGT = false;
				cv::Vec3b intensity = gtframe.at<cv::Vec3b>(r, c);
				if ((intensity.val[0] == 255) && (intensity.val[1] == 0) && (intensity.val[2] == 255))
				{
					pointInGT = true;
				}

				double pBBarea = 0.0;
				for (int i = 0; i < 4; i++)
				{
					pBBarea += GetTriangleAreaFromVertices(cv::Point(r, c), boudingBoxVertices[i], boudingBoxVertices[((i + 1) % 4)]);
				}
				bool pointInBB = (pBBarea <= boundingBoxArea) ? true : false;

				if ((pointInBB) && (pointInGT))
				{
					intersectionCount++;
					unionCount++;
				}
				else if ((pointInBB) || (pointInGT))
				{
					unionCount++;
				}
			}
		}
	}
	float iou = 0.0f;
	if (unionCount > 0)
	{
		iou = (float)intersectionCount / (float)unionCount;
	}
	return iou;
}

typedef enum
{
	RunMode_Image,
	RunMode_ImageKPI_Sligo,
	RunMode_ImageKPI_KITTI,
	RunMode_Video,
	RunMode_Tune_Sligo,
	RunMode_Tune_KITTI,
} ERunMode;

int main(int argc, char* argv[])
{
	//.\CppLaneDetection.exe archive\\data_road_224\\training\\image_2 223 2 10 223 90 110 140 110 170 223 archive\\data_road_224\\training\\gt_image_2\\gt_image_224_223.png
	std::cout << "CPP Lane Detection!" << std::endl;
	int fID = 223;//19131//29741
	std::string subfolder = "archive\\data_road_224\\training\\image_2";
	std::string kittiGTPath = "archive\\data_road_224\\training\\gt_image_2\\gt_image_224_223.png";
	ERunMode mode = RunMode_ImageKPI_KITTI;
	int gt1X = 400;
	int gt1Y = 700;
	int gt2X = 800;
	int gt2Y = 300;
	int gt3X = 1000;
	int gt3Y = 300;
	int gt4X = 1700;
	int gt4Y = 700;

	int roi1X = 10;
	int roi1Y = 223;
	int roi2X = 90;
	int roi2Y = 110;
	int roi3X = 140;
	int roi3Y = 110;
	int roi4X = 170;
	int roi4Y = 223;
	
	if (argc >= 4)
	{
		subfolder = argv[1];
		std::cout << "*** " << subfolder << "\\" << fID << ".jpg ***" << std::endl;
		fID = std::stoi(argv[2]);
		mode = (ERunMode)std::stoi(argv[3]);
		if (argc >= 12)
		{
			roi1X = std::stoi(argv[4]);
			roi1Y = std::stoi(argv[5]);
			roi2X = std::stoi(argv[6]);
			roi2Y = std::stoi(argv[7]);
			roi3X = std::stoi(argv[8]);
			roi3Y = std::stoi(argv[9]);
			roi4X = std::stoi(argv[10]);
			roi4Y = std::stoi(argv[11]);
			std::cout << "ROI: " << std::endl;
			std::cout << "ROI1(" << roi1X << ", " << roi1Y << ")" << std::endl;
			std::cout << "ROI2(" << roi2X << ", " << roi2Y << ")" << std::endl;
			std::cout << "ROI3(" << roi3X << ", " << roi3Y << ")" << std::endl;
			std::cout << "ROI4(" << roi4X << ", " << roi4Y << ")" << std::endl;

			if (argc == 20)
			{
				gt1X = std::stoi(argv[12]);
				gt1Y = std::stoi(argv[13]);
				gt2X = std::stoi(argv[14]);
				gt2Y = std::stoi(argv[15]);
				gt3X = std::stoi(argv[16]);
				gt3Y = std::stoi(argv[17]);
				gt4X = std::stoi(argv[18]);
				gt4Y = std::stoi(argv[19]);
				
				std::cout << "Ground Truth: " << std::endl;
				std::cout << "P1(" << gt1X << ", " << gt1Y << ")" << std::endl;
				std::cout << "P2(" << gt2X << ", " << gt2Y << ")" << std::endl;
				std::cout << "P3(" << gt3X << ", " << gt3Y << ")" << std::endl;
				std::cout << "P4(" << gt4X << ", " << gt4Y << ")" << std::endl;
			}
			else if (argc == 13)
			{
				kittiGTPath = argv[12];
			}
		}
	}
	
	cv::Mat roiMask;
	int bbY = 0;
	std::vector<cv::Point> boudingBoxVertices;
	std::vector<cv::Point> roiVertices;
	std::vector<cv::Point> groundTruthVertices;

	roiVertices.push_back(cv::Point(roi1X, roi1Y));
	roiVertices.push_back(cv::Point(roi2X, roi2Y));
	roiVertices.push_back(cv::Point(roi3X, roi3Y));
	roiVertices.push_back(cv::Point(roi4X, roi4Y));

	groundTruthVertices.push_back(cv::Point(gt1X, gt1Y));
	groundTruthVertices.push_back(cv::Point(gt2X, gt2Y));
	groundTruthVertices.push_back(cv::Point(gt3X, gt3Y));
	groundTruthVertices.push_back(cv::Point(gt4X, gt4Y));

	// Set some default hyperparameters
	LaneDetectionHyperparameters hyperparams;
	hyperparams.cannyHyperparams.t1 = 100.0f;
	hyperparams.cannyHyperparams.t2 = 120.0f;
	hyperparams.houghHyperparams.rho = 1.0f;
	hyperparams.houghHyperparams.theta = 1.0f;
	hyperparams.houghHyperparams.threshold = 20;
	hyperparams.houghHyperparams.minLineLength = 20.0f;
	hyperparams.houghHyperparams.maxLineGap = 50.0f;

	if ((mode == RunMode_Image) || (mode == RunMode_ImageKPI_Sligo) || (mode == RunMode_ImageKPI_KITTI))
	{
		cv::Mat frame = cv::imread(WORKING_DIRECTORY + subfolder + "\\image" + std::to_string(fID) + ".png");
		std::cout << WORKING_DIRECTORY + subfolder + "\\image" + std::to_string(fID) + ".png" << std::endl;
		bbY = CreateROIMask(roiMask, frame.cols, frame.rows, roiVertices);
		if (SAVE_EVERYTHING)
		{
			cv::imwrite(WORKING_DIRECTORY + "roiMask.jpg", roiMask);
		}
		cv::Mat colorMaskedFrame;
		ExtractYellowAndWhite(frame, colorMaskedFrame);
		cv::Mat outFrame;
		DoLaneDetection(frame, colorMaskedFrame, roiMask, bbY, fID, hyperparams, outFrame, boudingBoxVertices);
		cv::imwrite(WORKING_DIRECTORY + "out_frame_" + std::to_string(fID) + ".jpg", outFrame);

		if(mode == RunMode_ImageKPI_Sligo)
		{
			std::cout << "IoU: " << CalculateIoU(frame.cols, frame.rows, boudingBoxVertices, groundTruthVertices) << std::endl;
		}
		else if (mode == RunMode_ImageKPI_KITTI)
		{
			cv::Mat gtframe = cv::imread(WORKING_DIRECTORY + kittiGTPath);
			std::cout << "IoU: " << CalculateIoU(gtframe, boudingBoxVertices) << std::endl;
		}
	}
	else if(mode == RunMode_Video)
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
		int frameLimit = 40000;
		while (capture.read(frame))
		{
			if (roiMaskInitDone == false)
			{
				bbY = CreateROIMask(roiMask, frame.cols, frame.rows, roiVertices);
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
	else if ((mode == RunMode_Tune_KITTI) || (mode == RunMode_Tune_Sligo))
	{
		cv::Mat gtframe = cv::imread(WORKING_DIRECTORY + kittiGTPath);
		for (double cannyBaseThreshold = 10.0; cannyBaseThreshold < 240.0; cannyBaseThreshold += 20.0)
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
								LaneDetectionHyperparameters hyperparams;
								hyperparams.cannyHyperparams.t1 = cannyBaseThreshold;
								hyperparams.cannyHyperparams.t2 = cannyBaseThreshold * 1.2;
								hyperparams.houghHyperparams.rho = rho;
								hyperparams.houghHyperparams.theta = (double)theta;
								hyperparams.houghHyperparams.threshold = threshold;
								hyperparams.houghHyperparams.minLineLength = minLineLength;
								hyperparams.houghHyperparams.maxLineGap = maxLineGap;

								// Print the hyperparameters
								std::cout << "Hyperparameters: Canny T1: " << hyperparams.cannyHyperparams.t1 <<
									"Canny T2: " << hyperparams.cannyHyperparams.t2 <<
									"Hough Rho: " << hyperparams.houghHyperparams.rho <<
									"Hough Theta: " << hyperparams.houghHyperparams.theta <<
									"Hough Threshold: " << hyperparams.houghHyperparams.threshold <<
									"Hough Min Line Len: " << hyperparams.houghHyperparams.minLineLength <<
									"Hough Max Line Gap: " << hyperparams.houghHyperparams.maxLineGap << std::endl;

								cv::Mat frame = cv::imread(WORKING_DIRECTORY + subfolder + "\\image" + std::to_string(fID) + ".jpg");
								bbY = CreateROIMask(roiMask, frame.cols, frame.rows, roiVertices);
								if (SAVE_EVERYTHING)
								{
									cv::imwrite(WORKING_DIRECTORY + "roiMask.jpg", roiMask);
								}
								cv::Mat colorMaskedFrame;
								ExtractYellowAndWhite(frame, colorMaskedFrame);
								cv::Mat outFrame;
								DoLaneDetection(frame, colorMaskedFrame, roiMask, bbY, fID, hyperparams, outFrame, boudingBoxVertices);
								cv::imwrite(WORKING_DIRECTORY + "out_frame_" + std::to_string(fID) + ".jpg", outFrame);

								if (mode == RunMode_Tune_Sligo)
								{
									std::cout << "IoU: " << CalculateIoU(frame.cols, frame.rows, boudingBoxVertices, groundTruthVertices) << std::endl;
								}
								else
								{
									std::cout << "IoU: " << CalculateIoU(gtframe, boudingBoxVertices) << std::endl;
								}
							}
						}
					}
				}
			}
		}
	}

	return 0;
}