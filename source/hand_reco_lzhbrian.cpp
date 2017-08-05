/*
 * hand_reco_lzhbrian.cpp
 * @Author: lzhbrian
 * @Date:   2017-05-23
 * @Last modified by:   lzhbrian
 * @Last modified time: 2017-05-23
 * @Note: DIP Course work
 */

/*!
 * Usage:
 * To compile
 *   g++ source/hand_reco_lzhbrian.cpp -I/path/to/opencv3/include -L/path/to/opencv3/lib -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_objdetect -lopencv_imgproc -lopencv_imgcodecs -o hand_reco
 * To run
 *   ./hand_reco
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
#define CONVEX_HULL_CLUSTER_THRESHOLD 30
#define CONVEX_DEFECT_DEPTH_THRESHOLD 30

using namespace cv;
using namespace std;

void clusterConvexHullPoints(vector<vector<Point> > &hull,
                             vector<vector<int> >   &hull_idx,
                             vector<vector<Point> > &clustered_hull,
                             vector<vector<int> >   &clustered_hull_idx,
                             int largest_contour_index,
                             double threshold
                            );

int isLiveOrNot(vector<Point> v1, vector<Point> v2);

int main()
{

    /* init */
    Mat frame;
    Mat gray;
    VideoCapture capture;
    double fps = 100; // FPS

    /* open camera */
    capture.open(0); // open the default camera
    if (!capture.isOpened())
    {
        cout << "Camera Open Failed ..." << endl;
        return 0;
    }

    /* for my mac, default: width=1280, height=720 */
    // cout << "Width: "  << capture.get(CV_CAP_PROP_FRAME_WIDTH)  << endl;
    // cout << "Height: " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
    double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    double ROI_x = 600. / 1000. * (width / DOWN_SAMPLE_RATIO);
    double ROI_y = 250. / 1000. * (height / DOWN_SAMPLE_RATIO);
    double ROI_width = 350. / 1000. * (width / DOWN_SAMPLE_RATIO);
    double ROI_height = 650. / 1000. * (height / DOWN_SAMPLE_RATIO);
    cout << "ROI stat: x=" << ROI_x << " y=" << ROI_y << " width=" << ROI_width << " height=" << ROI_height << " \n";

    vector<Point> last_finger_ends;
    vector<Point> this_finger_ends;
    int live = 0;

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

            /* Find Largest Contour */
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            findContours( gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            int largest_contour_index = 0;
            int largest_area = 0;
            Rect bounding_rect;
            for( int i = 0; i < contours.size(); i++ )
            {
                //  Find the area of contour
                double area = contourArea(contours[i], false); 
                if(area > largest_area){
                    largest_area = area; 
                    // cout << i << " area  " << area << endl;
                    largest_contour_index = i; // Store the index of largest contour
                }
            }
            // drawContours( region_of_interest, contours, largest_contour_index, Scalar(255, 255, 0));
            // imshow("Largest contour", region_of_interest);

            /* Find Convex Hull */
            vector<vector<Point> > hull(contours.size());
            vector<vector<int> > hull_idx(contours.size());
            convexHull(contours[largest_contour_index], hull[largest_contour_index], false);
            convexHull(contours[largest_contour_index], hull_idx[largest_contour_index], false);

            // drawContours( region_of_interest, hull, largest_contour_index, Scalar(255, 255, 0), 2);
            // imshow("Convex Hull", region_of_interest);

            /* Cluster the Convex Hull points */
            vector<vector<Point> > clustered_hull(contours.size());
            vector<vector<int> > clustered_hull_idx(contours.size());
            clusterConvexHullPoints(hull, hull_idx, clustered_hull, clustered_hull_idx, largest_contour_index, CONVEX_HULL_CLUSTER_THRESHOLD);

            // Mat region_of_interest_1 = region_of_interest;

            // for(int i = 0; i < hull[largest_contour_index].size(); i++){
            //     circle(region_of_interest, hull[largest_contour_index][i], 2, Scalar(0, 0, 255), 2);
            // }
            // imshow("Before cluster", region_of_interest);

            // for(int i = 0; i < clustered_hull[largest_contour_index].size(); i++){
            //     circle(region_of_interest_1, clustered_hull[largest_contour_index][i], 2, Scalar(255, 255, 0), 2);
            // }
            // imshow("After cluster", region_of_interest_1);

            /* Find convexityDefects */
            vector<Vec4i> defects;
            vector<Point> tmp_finger_ends; // pass to isLiveOrNot()
            int valid_finger_end_points = 0;
            convexityDefects(contours[largest_contour_index], clustered_hull_idx[largest_contour_index], defects);
            for(int j = 0; j < defects.size(); j ++)
            {
                const Vec4i& v = defects[j];
                float depth = v[3] / 256;
                if (depth > CONVEX_DEFECT_DEPTH_THRESHOLD) //  filter defects by depth
                {
                    valid_finger_end_points += 1;
                    int startidx = v[0]; Point start_point(contours[largest_contour_index][startidx]);
                    int endidx = v[1]; Point end_point(contours[largest_contour_index][endidx]);
                    int faridx = v[2]; Point far_point(contours[largest_contour_index][faridx]);
                    circle(region_of_interest, start_point, 4, Scalar(0, 255, 255), 3); // Convex Hull Points
                    tmp_finger_ends.push_back(start_point);

                    line(region_of_interest, start_point, end_point, Scalar(0, 0, 255), 2); // outline
                    line(region_of_interest, start_point, far_point, Scalar(255, 0, 0), 2);
                    line(region_of_interest, end_point, far_point, Scalar(255, 0, 0), 2);

                    circle(region_of_interest, far_point, 4, Scalar(0, 255, 0), 2); // Convex Defect Points
                }
            }

            /* judge if its alive or not */
            // cerr << valid_finger_end_points << " ";
            if(live == 1){
                live = 0;
                // getchar();
            }
            if (valid_finger_end_points == 4)
            {
                last_finger_ends = this_finger_ends;
                this_finger_ends = tmp_finger_ends;
                /* Figure if it's live or not (zuotianyou) */
                int live = 0;
                if (last_finger_ends.size() && this_finger_ends.size())
                    live = isLiveOrNot( last_finger_ends, this_finger_ends );
                
                if(live) {
                   cout << "ALIVE !!!" << endl;  
                    live = 1;
                    putText(region_of_interest, "ALIVE" , Point(10, int(region_of_interest.rows*0.9)), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 255), 3);  
                }
            }
            imshow("Hand", region_of_interest); // show this frame

            /* Draw ROI on original frame*/
            rectangle(frame, cvRect(ROI_x, ROI_y, ROI_width, ROI_height), 
                Scalar(0, 255, 255), 3, 8 );
            imshow("Camera", frame); // show this frame
        }

        /* enter 'q' to exit */
        if (char(waitKey(fps)) == 'q') break;
    }

    capture.release();
    return 0;

}




/*!
 * \brief Cluster the Convex Hull Points
 * \param hull Input vector Hull points
 * \param hull_idx Input Vector Hull index of points
 * \param clustered_hull Output vector Clustered Hull points
 * \param clustered_hull_idx Output vector Clustered Hull index of points
 * \param largest_contour_index 
 * \param threshold Cluster distance threshols
 */
void clusterConvexHullPoints(vector<vector<Point> > &hull,
                             vector<vector<int> >   &hull_idx,
                             vector<vector<Point> > &clustered_hull,
                             vector<vector<int> >   &clustered_hull_idx,
                             int largest_contour_index,
                             double threshold
                            )
{
    Point last_point = hull[largest_contour_index][0];
    int last_point_idx = hull_idx[largest_contour_index][0];
    Point this_point = hull[largest_contour_index][0];
    int this_point_idx = hull_idx[largest_contour_index][0];
    vector<Point> point_set;
    vector<int> point_idx_set;
    for(int i = 0; i < hull[largest_contour_index].size(); i++)
    {
        last_point = this_point;
        last_point_idx = this_point_idx;
        this_point = hull[largest_contour_index][i];
        this_point_idx = hull_idx[largest_contour_index][i];
        if (norm(this_point - last_point) < threshold) {
            point_set.push_back(this_point);
            point_idx_set.push_back(this_point_idx);
        } else {
            clustered_hull[largest_contour_index].push_back(point_set[point_set.size() * 0]);
            clustered_hull_idx[largest_contour_index].push_back(point_idx_set[point_idx_set.size() * 0]);
            point_set.erase(point_set.begin(), point_set.end());
            point_idx_set.erase(point_idx_set.begin(), point_idx_set.end());
            point_set.push_back(this_point);
            point_idx_set.push_back(this_point_idx);
        }
    }
    if(point_set.size() != 0)
    {
        clustered_hull[largest_contour_index].push_back(point_set[point_set.size() * 0]);
        clustered_hull_idx[largest_contour_index].push_back(point_idx_set[point_idx_set.size() * 0]);
    }
}


int isLiveOrNot(vector<Point> v1, vector<Point> v2)
{
    for (int i = 0; i < v1.size(); i++)
        cout << "v1: " << v1[i] << endl;
    for (int i = 0; i < v2.size(); i++)
        cout << "v2: " << v2[i] << endl;

    cout << "Call isLiveOrNot ... \n";
    Point p1 = v1[0];
    Point p2 = v1[1];
    Point p3 = v1[2];
    Point p4 = v1[3];
    Point rp1 = v2[0];
    Point rp2 = v2[1];
    Point rp3 = v2[2];
    Point rp4 = v2[3];

    //rank
    double l1, l2, l3, rl1, rl2, rl3, movement, l, rl;
    l1 = norm(p1 - p2);
    l2 = norm(p2 - p3);
    l3 = norm(p3 - p4);

    rl1 = norm(rp1 - rp2);
    rl2 = norm(rp2 - rp3);
    rl3 = norm(rp3 - rp4);

    l = l1 + l2 + l3;
    rl = rl1 + rl2 + rl3;
    l1 = l1 / l; l2 = l2 / l; l3 = l3 / l;
    rl1 = rl1 / rl; rl2 = rl2 / rl; rl3 = rl3 / rl;
    movement = abs(l1-rl1) + abs(l2-rl2) + abs(l3-rl3);
    cout << "movement: " << movement << endl;
    if(movement > 0.12 && movement < 0.2)        
        return 1;
    else
        return 0;
}

