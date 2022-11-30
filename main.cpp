#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
int main()
{
    std::string image_folder = "/Users/ernsjus/Dev/openMPITest/images";

    std::string image_path = image_folder + "/animal/1.kitten_small.jpg";
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);

    int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}
