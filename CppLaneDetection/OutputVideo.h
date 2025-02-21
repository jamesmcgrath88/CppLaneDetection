#pragma once

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <iostream>

class OutputVideo
{
	cv::VideoWriter writer;
	double fps = 30.0;

public:
	bool WriteFrameToOutputVideo(cv::Mat frameToWrite);
	bool SaveOutputVideo(void);
};