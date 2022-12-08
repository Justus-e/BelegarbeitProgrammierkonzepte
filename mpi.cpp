//
// Created by Ernst, Justus (415) on 30.11.22.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <mpi.h>

using namespace cv;

Mat rgbToGray(Mat img) {
    Mat newImg = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            Vec3b pixel = img.at<Vec3b>(i, j);
            // B, G, R
            double gray = pixel[0] * 0.07 + pixel[1] * 0.72 + pixel[2] * 0.21;

            newImg.at<Vec<uchar, 1>>(i, j)[0] = gray;

        }
    }
    return newImg;
}


int main(int argc, char** argv) {

    int rank, size;

    // image properties:
    int image_properties[4];

    //image
    Mat fullImg;

    std::string image_folder = "/Users/ernsjus/Dev/openMPITest/images";
    std::string image_path = image_folder + "/nature/4.nature_mega.jpeg";

    // init MPI
    MPI_Init( &argc, &argv );

    // get the size and rank
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    // read in the image, then divide it in equal sized stripes for each process and covert it to grayscale. combine the stripes so that each process has the full image




    // finalize MPI
    MPI_Finalize();

    return 0;
}