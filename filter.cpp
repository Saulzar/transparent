#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <iostream>

using namespace boost::filesystem;

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

inline void display(cv::Mat const &image) {
  cv::imshow("image", image);
  cv::waitKey(0);
}


const char* keys =
  { "{help h||}{@input||input path}" };


bool clipped(cv::Mat const& alpha) {

  cv::Mat thresh = alpha > 50;

  int top = cv::countNonZero(alpha.row(0));
  int bot = cv::countNonZero(alpha.row(alpha.rows - 1));

  int left = cv::countNonZero(alpha.col(0));
  int right = cv::countNonZero(alpha.col(alpha.cols - 1));


  return (top > alpha.cols * 0.05 || bot > alpha.rows * 0.05
        || left > alpha.cols * 0.05 || right > alpha.rows * 0.05);
}

bool bestComponent(cv::Mat1b const &alpha, cv::Mat& mask) {
  int totalArea = alpha.cols * alpha.rows;

  cv::Mat1i labels;
  int n = cv::connectedComponents(alpha > 200, labels, 8, CV_32S);

  int best = -1;
  float bestArea = 0;



  for(int i = 0; i < n; ++i) {
    cv::Mat label = alpha.mul(labels == i, 1);
    float area = cv::countNonZero(label);

    if(!clipped(label) && area > float(totalArea) * 0.1) {
      vector<vector<Point> > contours;
      findContours( label.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      if(contours.size() == 1 && area > bestArea) {

        vector<Point> approx;
        approxPolyDP(contours[0], approx, 4.0, true);

        if(!isContourConvex(approx)) {

          best = i;
          bestArea = area;
        }
      }
    }
  }

  if(best >= 0) {
    mask = (labels == best) & 255;
    return true;
  }

  return false;
}


inline cv::Mat1b getAlpha(cv::Mat4b const &image) {
  std::vector<cv::Mat1b> channels;
  cv::split(image, channels);

  return channels[3];
}

inline cv::Mat3b getRgb(cv::Mat4b const &image) {
  std::vector<cv::Mat1b> channels;
  cv::split(image, channels);

  cv::Mat3b rgb;
  merge(std::vector<Mat1b> {channels[0], channels[1], channels[2]}, rgb);

  return rgb;
}




inline cv::Mat4b replaceAlpha(cv::Mat4b const &image, cv::Mat1b const &alpha) {
  std::vector<cv::Mat1b> channels;
  cv::split(image, channels);

  channels[3] = alpha;

  cv::Mat result;
  cv::merge(channels, result);

  return result;

}


bool keepImage(cv::Mat4b &result, std::string const &path) {

  cv::Mat4b img = imread(path.c_str(), cv::IMREAD_UNCHANGED);
  if(img.empty() || img.channels() != 4) return false;

  cv::Mat1b alpha = getAlpha(img);

  cv::Mat mask;
  if(bestComponent(alpha, mask)) {
    cv::Mat1b r;

    cv::GaussianBlur(mask, mask, cv::Size(9, 9), 0);
    r = alpha.mul(mask + mask, 1.0/255);

    display(mask);

    result = replaceAlpha(img, r);
    return true;
  }

  return false;
}




cv::Mat4b trimAlpha(cv::Mat4b const &image) {
  cv::Mat1b alpha = getAlpha(image);

  vector<vector<Point> > contours;
  findContours(alpha, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  RotatedRect rect = cv::minAreaRect(contours[0]);
  Size size = rect.size;
  float angle = rect.angle;
  if(angle < -45.) {
    angle += 90.0;
    swap(size.width, size.height);
  }

  Mat m = getRotationMatrix2D(rect.center, angle, 1.0);
  Mat trimmed;

  warpAffine(image, trimmed, m, image.size(), INTER_CUBIC);

  return trimmed;
}

cv::Mat checkers(int w, int h, int blockSize) {

  Mat3b img(h, w);
  img.setTo(cv::Scalar(127, 127, 127));

  for(int i = 0; i < w; i += blockSize) {
    for(int j = 0; j < h; j += blockSize) {

      if((i / blockSize + j / blockSize) & 1) {
        auto cols = Range(j, min(h, j + blockSize - 1));
        auto rows = Range(i, min(w, i + blockSize - 1));

        img(cols, rows).setTo(cv::Scalar(192, 192, 192));
      }
    }
  }

  return img;
}

inline cv::Mat dup(cv::Mat1b const &a, int n) {
  std::vector<cv::Mat1b> v(n, a);

  cv::Mat m;
  cv::merge(v, m);

  return m;
}

cv::Mat alphaBlend(const Mat4b& image, Mat3b const& dest) {

  cv::Mat3b alpha = dup(getAlpha(image), 3);
  cv::Mat3b rgb = getRgb(image);

  cv::Mat blend = rgb.mul(alpha, 1.0/255) + dest.mul(cv::Scalar(255, 255, 255) - alpha, 1.0/255);
  return blend;
}


int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    string inputPath = parser.get<string>(0);

    cv::Mat4b image;
    for(auto& entry : boost::make_iterator_range(directory_iterator(inputPath), {})) {
      if( !is_regular_file( entry.status() ) ) continue;

      std::string path = entry.path().c_str();
      if(keepImage(image, path)) {

        cv::Mat bg = checkers(image.cols, image.rows, 12);

        image = trimAlpha(image);
        cv::Mat out = alphaBlend(image, bg);

        display(out);

      }
    }

    return 0;
}
