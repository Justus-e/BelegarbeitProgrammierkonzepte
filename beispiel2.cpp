#include <iostream>
#include <string>

#include <mpi.h>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    int rank, size;

    // the full image:
    cv::Mat full_image;

    // image properties:
    int image_properties[4];

    // init MPI
    MPI_Init( &argc, &argv );

    // get the size and rank
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    // load the image ONLY in the master process #0:
    if ( rank == 0 ) {
        full_image = cv::imread( "D:\\cat.jpg", cv::IMREAD_UNCHANGED );

        // get the properties of the image, to send to other processes later:
        image_properties[0] = full_image.cols; // width
        image_properties[1] = full_image.rows / size; // height, divide it by number of processes
        image_properties[2] = full_image.type(); // image type (in this case: CV_8UC3 - 3 channel, 8 bit )
        image_properties[3] = full_image.channels(); // number of channels (here: 3)
    }

    // wait for it to finish:
    MPI_Barrier( MPI_COMM_WORLD );

    // now broadcast the image properties from process #0 to all others:
    // the 'image_properties' array is only initialized in process #0!
    // that's why your IDE might show a warning here
    MPI_Bcast( image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD );

    // now all processes have these properties, initialize the "partial" image in each process (height, width, type)
    cv::Mat part_image = cv::Mat( image_properties[1], image_properties[0], image_properties[2] );

    // wait for all to finish:
    MPI_Barrier( MPI_COMM_WORLD );

    // now we can cut the full_image into parts and send the parts to each process

    // first, the number of bytes to send: (Height * Width * Channels)
    int send_size = image_properties[1] * image_properties[0] * image_properties[3];

    // from process #0 scatter to all others:
    // the cv::Mat.data is a pointer to the raw image data (B1,G1,R1,B2,G2,R2,.....)
    MPI_Scatter( full_image.data, send_size, MPI_UNSIGNED_CHAR, // unsigned char = unsigned 8-bit (byte)
                 part_image.data, send_size, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD ); // from process #0

    // of course, you can Bcast the full image instead of scattering it, or use Send/Recv (set the correct height!)

    // now all the PROCESSES have their own copy of the 'part_image' image, which contains
    // a horizontal slice of the image
    // we can do something with it...

    cv::imshow( "image in process #" + std::to_string( rank ), part_image );

    cv::waitKey( 0 ); // will need to press a key in EACH process...
    cv::destroyAllWindows();

    // save? (each process will save their own image, with different names)
    cv::imwrite( "image_slice_" + std::to_string( rank ) + ".jpg", part_image );

    // you can also, after doing something with the parts, MPI_Gather it back into process #0:

    MPI_Gather( part_image.data, send_size, MPI_UNSIGNED_CHAR,
                full_image.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD );

    if ( rank == 0 ) {
        std::cout << "Process #0 received the gathered image" << std::endl;

        cv::imshow( "gathered image", full_image );

        cv::waitKey( 0 ); // will need to press a key in EACH process...
        cv::destroyAllWindows();
    }

    // finalize MPI
    MPI_Finalize();
}