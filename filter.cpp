#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat colorize(cv::Mat labelImage, int nLabels) {
  std::vector<Vec3b> colors(nLabels);
  colors[0] = Vec3b(0, 0, 0);//background
  for(int label = 1; label < nLabels; ++label) {
    colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
  }

  Mat dst(labelImage.size(), CV_8UC3);
  for(int r = 0; r < dst.rows; ++r){
    for(int c = 0; c < dst.cols; ++c){
      int label = labelImage.at<int>(r, c);
      Vec3b &pixel = dst.at<Vec3b>(r, c);
      pixel = colors[label];
    }
  }

  return dst;
}

const char* keys =
  { "{help h||}{@image||image for converting to a grayscale}" };


bool clipped(cv::Mat alpha) {

  double top = cv::sum(alpha.row(0))[0];
  double bot = cv::sum(alpha.row(alpha.rows - 1))[0];

  double left = cv::sum(alpha.col(0))[0];
  double right = cv::sum(alpha.col(alpha.cols - 1))[0];

  return (top > alpha.cols * 0.25 || bot > alpha.rows * 0.25
        || left > alpha.cols * 0.25 || right > alpha.rows * 0.25);
}

bool bestComponent(cv::Mat const &alpha, cv::Mat& mask) {
  cv::Mat1s stats;
  cv::Mat1d centroids;

  cv::Mat labels;
  int n = cv::connectedComponentsWithStats(alpha, labels, stats, centroids,  8);
  int totalArea = alpha.cols * alpha.rows;

  int best = -1;
  float bestArea = 0;

  for(int i = 0; i < n; ++i) {
    std::cout << i << " of " << n << std::endl;
    cv::Mat label = (labels == i) & 255;
    float area = stats(i, CC_STAT_AREA);

    if(area > float(totalArea) * 0.15) {
      vector<vector<Point> > contours;
      findContours( label.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      if(contours.size() == 1 && !isContourConvex(contours[0]) && area > bestArea) {
        best = i;
        bestArea = stats(i, CC_STAT_AREA);
      }
    }
  }

  if(best >= 0) {
    mask = (labels == best) & 255;
    return true;
  }

  return false;
}


int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);

    string inputImage = parser.get<string>(0);
    cv::Mat img = imread(inputImage.c_str(), IMREAD_UNCHANGED);

    if(img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return -1;
    }

    std::vector<cv::Mat1b> channels;
    cv::split(img, channels);

    cv::Mat1b alpha = channels[3] > 50;
    cv::Mat mask;

    if(!clipped(alpha) && bestComponent(alpha, mask)) {



    }




    return 0;
}
