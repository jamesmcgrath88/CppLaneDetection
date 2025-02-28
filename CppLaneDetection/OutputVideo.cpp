#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <iostream>
#include "OutputVideo.h"

OutputVideo::OutputVideo(const std::string workingDir)
{
	workingDirectory = workingDir;
}

bool OutputVideo::WriteFrameToOutputVideo(cv::Mat frameToWrite)
{
	bool res = true;

	if (!writer.isOpened())
	{
		bool isColor = (frameToWrite.type() == CV_8UC3);
		int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		std::string outputVideo(workingDirectory + "Singlecarriageway_out.mp4");
		writer.open(outputVideo, codec, fps, frameToWrite.size(), isColor);
	}

	writer.write(frameToWrite);

	return res;
}

bool OutputVideo::SaveOutputVideo(void)
{
	writer.release();
	return true;
}
