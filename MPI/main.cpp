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

void grayscaleBlur(int blur_strength, std::string image_folder, std::string image_path) {

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

    int rows_per_process[size];
    int sendsize_per_process[size];
    //int row_displacements[size];
    int sendsize_displacements[size];
    int stripe_height = height / size;
    int total_rows = 0;

    if (blur_strength > stripe_height) {
        std::cout << "no good\n";
    }

    for (int i = 0; i < size; ++i) {
        //row_displacements[i] = total_rows;
        sendsize_displacements[i] = total_rows * width * 3;
        rows_per_process[i] = stripe_height + (i < (height % size));
        sendsize_per_process[i] = rows_per_process[i] * width * 3;
        total_rows += rows_per_process[i];
    }

    cv::Mat part_image = cv::Mat(rows_per_process[rank], width, CV_8UC3);

    MPI_Scatterv(full_image.data, sendsize_per_process, sendsize_displacements, MPI_UNSIGNED_CHAR,
                 part_image.data, sendsize_per_process[rank], MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD); // from process #0

    cv::Mat gray_part = rgbToGray(part_image);

    gray_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < size; ++i) {
        sendsize_per_process[i] = sendsize_per_process[i] / 3;
        sendsize_displacements[i] = sendsize_displacements[i] / 3;
    }

    MPI_Gatherv(gray_part.data, sendsize_per_process[rank], MPI_UNSIGNED_CHAR,
                gray_image.data, sendsize_per_process, sendsize_displacements, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank > 0) {
        MPI_Send(gray_part.data, blur_strength * width, MPI_UNSIGNED_CHAR, rank - 1, 1, MPI_COMM_WORLD);
    }

    if (rank < size - 1) {
        cv::Mat padding = cv::Mat(blur_strength, width, CV_8UC1, cv::Scalar(0));
        MPI_Recv(padding.data, blur_strength * width, MPI_UNSIGNED_CHAR, rank + 1, 1, MPI_COMM_WORLD, nullptr);
        MPI_Send(gray_part.data + (sendsize_per_process[rank] - blur_strength * width), blur_strength * width,
                 MPI_UNSIGNED_CHAR, rank + 1, 2, MPI_COMM_WORLD);
        gray_part.push_back(padding);
    }

    if (rank > 0) {
        cv::Mat padding = cv::Mat(blur_strength, width, CV_8UC1, cv::Scalar(0));
        MPI_Recv(padding.data, blur_strength * width, MPI_UNSIGNED_CHAR, rank - 1, 2, MPI_COMM_WORLD, nullptr);
        padding.push_back(gray_part);

        gray_part = cv::Mat(padding);
    }

    int start_row = 0;
    if (rank > 0) {
        start_row = blur_strength;
    }
    int end_row = start_row + rows_per_process[rank];

    cv::Mat blur_part = blur(gray_part, start_row, end_row, blur_strength);

    if (rank == 0) {
        blurred_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    }

    MPI_Gatherv(blur_part.data, sendsize_per_process[rank], MPI_UNSIGNED_CHAR,
                blurred_image.data, sendsize_per_process, sendsize_displacements, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        imwrite(image_folder + "/output/gray.png", gray_image);
        imwrite(image_folder + "/output/gray&blur.png", blurred_image);
    }

}

int main(int argc, char **argv) {
    std::ofstream myFile("/Users/ernsjus/Dev/parallel/mpi_8_natureMega.csv");
    std::string image_folder = "/Users/ernsjus/Dev/parallel/images";
    std::string image_path = image_folder + "/nature/4.nature_mega.jpeg";
    int blur_strength = 10;

    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < 500; ++i) {
        double start, end;
        if (rank == 0) {
            std::cout << i << "\n";
            start = MPI_Wtime();
        }

        grayscaleBlur(blur_strength, image_folder, image_path);

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