#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>

void applyBlurToPixel(cv::Mat &img, cv::Mat &newImg, int i, int j) {
    bool isLeftSide = j == 0;
    bool isRightSide = j == img.cols - 1;
    bool isTop = i == 0;
    bool isBottom = i == img.rows - 1;

    bool isCorner = (isLeftSide and isTop) or (isRightSide and isTop) or (isLeftSide and isBottom) or
                    (isRightSide and isBottom);
    bool isEdge = isTop xor isBottom xor isRightSide xor isLeftSide;

    double divisor;
    if (isCorner) {
        divisor = 4;
    } else if (isEdge) {
        divisor = 6;
    } else {
        divisor = 9;
    }

    double sum = 0;

    for (int l = -1; l < 2; ++l) {

        if (i + l < 0 or i + l > img.rows - 1) {
            continue;
        }

        for (int m = -1; m < 2; ++m) {

            if (j + m < 0 or j + m > img.cols - 1) {
                continue;
            }

            sum += img.at<cv::Vec<uchar, 1>>(i + l, j + m)[0];

        }
    }

    newImg.at<cv::Vec<uchar, 1>>(i, j)[0] = sum / divisor;
}

cv::Mat blur(cv::Mat &img) {
    cv::Mat newImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            applyBlurToPixel(img, newImg, i, j);

        }
    }
    return newImg;
}

cv::Mat rgbToGray(cv::Mat &img) {
    cv::Mat newImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            // B, G, R
            double gray = pixel[0] * 0.07 + pixel[1] * 0.72 + pixel[2] * 0.21;

            newImg.at<cv::Vec<uchar, 1>>(i, j)[0] = gray;

        }
    }
    return newImg;
}


int main() {

    std::ofstream myFile("/Users/ernsjus/Dev/openMPITest/seriell_haroldMedium.csv");


    for (int i = 0; i < 1000; ++i) {
        std::cout << i << "\n";

        double t0 = omp_get_wtime(); // start time

        std::string image_folder = "/Users/ernsjus/Dev/openMPITest/images";

        std::string image_path = image_folder + "/human/2.harold_medium.jpg";
        cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }

        cv::Mat grayImg = rgbToGray(img);
        cv::Mat bluredImg = blur(grayImg);

        imwrite(image_folder + "/output/gray.png", grayImg);
        imwrite(image_folder + "/output/gray&blur.png", bluredImg);

        double t1 = omp_get_wtime();  // end time
        myFile << t1 - t0 << "\n";

    }

    myFile.flush();
    myFile.close();


    return 0;
}