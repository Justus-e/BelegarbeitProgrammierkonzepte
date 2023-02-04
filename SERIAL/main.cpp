#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>

void applyBlurToPixel(cv::Mat &img, cv::Mat &newImg, int i, int j, int strength) {
    double divisor = 0;

    double sum = 0;

    for (int l = -strength; l <= strength; ++l) {

        if (i + l < 0 or i + l > img.rows - 1) {
            continue;
        }

        for (int m = -strength; m <= strength; ++m) {

            if (j + m < 0 or j + m > img.cols - 1) {
                continue;
            }

            sum += img.at<cv::Vec<uchar, 1>>(i + l, j + m)[0];
            divisor++;

        }
    }

    newImg.at<cv::Vec<uchar, 1>>(i, j)[0] = sum / divisor;
}

cv::Mat blur(cv::Mat &img, int strength) {
    cv::Mat newImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            applyBlurToPixel(img, newImg, i, j, strength);

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

    std::ofstream myFile("/Users/ernsjus/Dev/parallel/seriell_natureMega.csv");
    std::string image_folder = "/Users/ernsjus/Dev/parallel/images";
    std::string image_path = image_folder + "/nature/4.nature_mega.jpeg";
    int blur_strength = 10;


    for (int i = 0; i < 500; ++i) {
        std::cout << i << "\n";

        double t0 = omp_get_wtime(); // start time

        cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }

        cv::Mat grayImg = rgbToGray(img);
        cv::Mat blurredImg = blur(grayImg, blur_strength);

        imwrite(image_folder + "/output/gray.png", grayImg);
        imwrite(image_folder + "/output/gray&blur.png", blurredImg);

        double t1 = omp_get_wtime();  // end time
        myFile << t1 - t0 << "\n";

    }

    myFile.flush();
    myFile.close();

    return 0;
}