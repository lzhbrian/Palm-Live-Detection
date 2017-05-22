/*
 * hand_reco_lzhbrian.cpp
 * @Author: lzhbrian
 * @Date:   2017-05-23
 * @Last modified by:   lzhbrian
 * @Last modified time: 2017-05-23
 * @Note: Course work for DIP
 */

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

#define DOWN_SAMPLE_RATIO 1.5 // to down sample the captured image

using namespace cv;

int main()
{

    /* init */
    Mat frame;
    Mat gray;
    VideoCapture capture;
    double fps = 10; // FPS

    /* open camera */
    capture.open(0); // open the default camera
    if (!capture.isOpened())
    {
        std::cout << "Camera Open Failed ..." << std::endl;
        return 0;
    }

    /* for my mac, default: width=1280, height=720 */
    // std::cout << "Width: "  << capture.get(CV_CAP_PROP_FRAME_WIDTH)  << std::endl;
    // std::cout << "Height: " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
    double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    double ROI_x = 600. / 1000. * (width / DOWN_SAMPLE_RATIO);
    double ROI_y = 300. / 1000. * (height / DOWN_SAMPLE_RATIO);
    double ROI_width = 300. / 1000. * (width / DOWN_SAMPLE_RATIO);
    double ROI_height = 600. / 1000. * (height / DOWN_SAMPLE_RATIO);

    std::cout << "ROI stat: x=" << ROI_x << " y=" << ROI_y << " width=" << ROI_width << " height=" << ROI_height << " \n";

    /* read every frame */
    while (true) {
        capture >> frame;

        if (!frame.empty()) // process
        {
            /* resize, flip, get region_of_interest */
            resize(frame, frame, Size(), 1.0/DOWN_SAMPLE_RATIO, 1.0/DOWN_SAMPLE_RATIO);
            flip(frame, frame, 1); // flip to be like mirror
            Mat region_of_interest = frame( Rect(ROI_x, ROI_y, ROI_width, ROI_height) );

            /* Convert to gray, Blur, 2-val */
            cvtColor(region_of_interest, gray, cv::COLOR_BGR2GRAY); // convert to grayscale
            // GaussianBlur( gray, gray, Size( 7, 7 ), 0, 0 ); // gaussian blur
            blur( gray, gray, Size( 12, 12 ), Point(-1,-1)); // homogenous blur
            threshold(gray, gray, 0, 255, (CV_THRESH_BINARY_INV + CV_THRESH_OTSU)); // 2-val
            imshow("Gray", gray);

            /* Figure if it's hand or not (zhangyu) */
            // int hand = 0;
            // hand = isHandOrNot( ... );
            // if(!hand) continue;

            /* Find contour */
            std::vector<std::vector<Point> > contours;
            std::vector<Vec4i> hierarchy;
            findContours( gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            
            /* Find Largest Contour */
            int largest_contour_index = 0;
            int largest_area = 0;
            Rect bounding_rect;
            for( int i = 0; i< contours.size(); i++ )
            {
                //  Find the area of contour
                double area = contourArea( contours[i],false); 
                if(area > largest_area){
                    largest_area = area; 
                    // std::cout << i << " area  " << area << std::endl;
                    largest_contour_index = i; // Store the index of largest contour
                }
            }
            Scalar color(255,255,255);
            drawContours(region_of_interest, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy);
            imshow("Contours", region_of_interest); // show this frame

            /* Get points in Contour */



            /* Figure if it's live or not (zuotianyou) */
            // int live = 0;
            // live = isLiveOrNot( ... );
            // if(live) std::cout << "ALIVE !!!" << std::endl;

            /* Draw ROI on original frame*/
            rectangle(frame, cvRect(ROI_x, ROI_y, ROI_width, ROI_height), 
                Scalar(0, 255, 255), 3, 8 );
            imshow("Camera", frame); // show this frame
            imshow("Hand", region_of_interest); // show this frame
        }

        /* enter 'q' to exit */
        if (char(waitKey(fps)) == 'q') break;
    }

    capture.release();
    return 0;

}