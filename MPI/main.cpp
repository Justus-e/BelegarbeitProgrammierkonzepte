#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <fstream>

cv::Mat rgbToGray(cv::Mat img) {
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

void applyBlurToPixel(const cv::Mat &img, cv::Mat &newImg, int i, int j, int strength, int start_row) {

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

    newImg.at<cv::Vec<uchar, 1>>(i - start_row, j)[0] = sum / divisor;
}

cv::Mat blur(cv::Mat &img, int start_row, int end_row, int strength) {
    cv::Mat newImg = cv::Mat(end_row - start_row, img.cols, CV_8UC1, cv::Scalar(0));

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            applyBlurToPixel(img, newImg, i, j, strength, start_row);

        }
    }
    return newImg;
}

void grayscaleBlur() {
    std::string image_folder = "/Users/ernsjus/Dev/parallel/images";
    std::string image_path = image_folder + "/nature/4.nature_mega.jpeg";

    int rank, size;

    cv::Mat full_image;
    cv::Mat gray_image;
    cv::Mat blurred_image;

    int width, height;

    // get the size and rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        full_image = cv::imread(image_path, cv::IMREAD_COLOR);
        width = full_image.cols;
        height = full_image.rows;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    gray_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    int stripe_height = height / size;

    int send_size = width * (stripe_height) * 3;

    cv::Mat part_image = cv::Mat(stripe_height, width, CV_8UC3);

    MPI_Scatter(full_image.data, send_size, MPI_UNSIGNED_CHAR,
                part_image.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD); // from process #0

    cv::Mat gray_part = rgbToGray(part_image);

    send_size = send_size / 3;

    MPI_Gather(gray_part.data, send_size, MPI_UNSIGNED_CHAR,
               gray_image.data, send_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    MPI_Bcast(gray_image.data, send_size * size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int start_row = rank * stripe_height;
    int end_row = start_row + stripe_height;

    cv::Mat blur_part = blur(gray_image, start_row, end_row, 10);

    if (rank == 0) {
        blurred_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    }

    MPI_Gather(blur_part.data, send_size, MPI_UNSIGNED_CHAR,
               blurred_image.data, send_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        imwrite(image_folder + "/output/gray.png", gray_image);
        imwrite(image_folder + "/output/gray&blur.png", blurred_image);
    }

}

int main(int argc, char **argv) {
    std::ofstream myFile("/Users/ernsjus/Dev/parallel/mpi_8_natureMega.csv");

    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < 1000; ++i) {
        double start, end;
        if (rank == 0) {
            std::cout << i << "\n";
            start = MPI_Wtime();
        }

        grayscaleBlur();

        if (rank == 0) {
            end = MPI_Wtime();
            myFile << end - start << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    myFile.flush();
    myFile.close();

    // finalize MPI
    MPI_Finalize();
}