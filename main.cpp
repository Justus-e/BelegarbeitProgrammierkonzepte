#include <iostream>

#include <mpi.h>

#include <opencv2/opencv.hpp>

cv::Mat rgbToGray(cv::Mat img) {
    cv::Mat newImg = cv::Mat(img.rows , img.cols, CV_8UC1, cv::Scalar(0));

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

void applyBlurToPixel(const cv::Mat& img, cv::Mat &newImg, int i, int j, int start_row) {
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

    int intensity = 1;

    for (int l = -intensity; l < intensity+1; ++l) {

        if (i + l < 0 or i + l > img.rows - 1) {
            continue;
        }

        for (int m = -intensity; m < intensity+1; ++m) {

            if (j + m < 0 or j + m > img.cols - 1) {
                continue;
            }

            sum += img.at<cv::Vec<uchar, 1>>(i + l, j + m)[0];

        }
    }

    newImg.at<cv::Vec<uchar, 1>>(i - start_row, j)[0] = sum / divisor;
}

cv::Mat blur(cv::Mat &img, int start_row, int end_row) {
    cv::Mat newImg = cv::Mat(end_row - start_row, img.cols, CV_8UC1, cv::Scalar(0));

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            applyBlurToPixel(img, newImg, i, j, start_row);

        }
    }
    return newImg;
}

int main(int argc, char** argv)
{
    std::string image_folder = "/Users/ernsjus/Dev/openMPITest/images";
    std::string image_path = image_folder + "/human/1.harold_small.jpg";

    int rank, size;

    cv::Mat full_image;
    cv::Mat gray_image;
    cv::Mat blurred_image;

    int width, height;

    // init MPI
    MPI_Init( &argc, &argv );

    // get the size and rank
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if (rank == 0) {
        full_image = cv::imread(image_path, cv::IMREAD_COLOR);
        width = full_image.cols;
        height = full_image.rows;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    gray_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    int send_size = width * (height / size) * 3;

    cv::Mat part_image = cv::Mat(height / size, width, CV_8UC3);

    MPI_Scatter( full_image.data, send_size, MPI_UNSIGNED_CHAR,
                 part_image.data, send_size, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD ); // from process #0

    cv::Mat gray_part = rgbToGray(part_image);

    send_size = send_size / 3;

    MPI_Gather( gray_part.data, send_size, MPI_UNSIGNED_CHAR,
                gray_image.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD );

    MPI_Bcast(gray_image.data, send_size * size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int stripe_height = height / size;
    int start_row = rank * stripe_height;
    int end_row = start_row + stripe_height;

    cv::Mat blur_part = blur(gray_image, start_row, end_row);

    if (rank == 0) {
        blurred_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    }

    MPI_Gather( blur_part.data, send_size, MPI_UNSIGNED_CHAR,
                blurred_image.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD );


    if (rank == 0) {
        imwrite(image_folder + "/output/gray.png", gray_image);
        imwrite(image_folder + "/output/gray&blur.png", blurred_image);
    }

    // finalize MPI
    MPI_Finalize();
}