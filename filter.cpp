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
  { "{help h||}{@image|../data/stuff.jpg|image for converting to a grayscale}" };

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);

    string inputImage = parser.get<string>(0);
    cv::Mat img = imread(inputImage.c_str(), 0);

    if(img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return -1;
    }

    std::vector<cv::Mat1b> channels;
    cv::split(img, channels);

    cv::Mat1b alpha = channels[0];

    cv::Mat2d centroids;
    cv::Mat1s stats;

    cv::Mat labels;
    int n = cv::connectedComponentsWithStats(alpha, labels, stats, centroids, 8);

    

    imshow( "image", colorize(labels, n) );


    waitKey(0);
    return 0;
}
