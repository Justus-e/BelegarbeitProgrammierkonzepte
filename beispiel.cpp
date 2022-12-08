// ONE IMAGE

#include <iostream>
#include <cstdio>
#include <omp.h>

#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{
  // read image
  cv::Mat image = cv::imread( "d:\\cat.jpg", cv::IMREAD_UNCHANGED );

  // display and wait for a key-press, then close the window
  cv::imshow( "image", image );
  int key = cv::waitKey( 0 );
  cv::destroyAllWindows();

  double t0 = omp_get_wtime(); // start time

  #pragma omp parallel for
  for ( int i = 0; i < image.rows; ++i ) {
    for ( int j = 0; j < image.cols; ++j ) {

      // get pixel at [i, j] as a <B,G,R> vector
      cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

      // extract the pixels as uchar (unsigned 8-bit) types (0..255)
      uchar b = pixel[0];
      uchar g = pixel[1];
      uchar r = pixel[2];

      // Note: this is actually the slowest way to extract a pixel in OpenCV
      // Using pointers like this:
      //   uchar* ptr = (uchar*) image.data; // get raw pointer to the image data
      //   ...
      //   for (...) {
      //       uchar* pixel = ptr + image.channels() * (i * image.cols + j);
      //       uchar b = *(pixel + 0); // Blue
      //       uchar g = *(pixel + 1); // Green
      //       uchar r = *(pixel + 2); // Red
      //       uchar a = *(pixel + 3); // (optional) if there is an Alpha channel
      //   }
      // is much faster

      uchar temp = r;
      r = b;
      b = temp;

      image.at<cv::Vec3b>( i, j ) = pixel;
      // or:
      // image.at<cv::Vec3b>( i, j ) = {r, g, b};
    }
  }
  double t1 = omp_get_wtime();  // end time

  std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;

  // display and wait for a key-press, then close the window
  cv::imshow( "image", image );
  key = cv::waitKey( 0 );
  cv::destroyAllWindows();
}