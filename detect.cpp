#include "opencv2/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace cv::face;
using namespace std;

//void detectAndDisplay( Mat frame );
void detectAndDisplay( Mat frame, Ptr<FaceRecognizer> modelF, Ptr<LBPHFaceRecognizer> modelL);

String face_cascade_name = "./data/haar_face.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";
int imageWidth = 92;
int imageHeight = 112;

int main( void )
{
    VideoCapture capture;
    Mat frame;
    Ptr<FaceRecognizer> modelF = createFisherFaceRecognizer();
    modelF->load("./data/train.gender.fisher");
    Ptr<LBPHFaceRecognizer> modelL = createLBPHFaceRecognizer();
    modelL->load("./data/train.age.lbph");

    if(!face_cascade.load(face_cascade_name))
    {
        printf("Error loading face cascade\n");
        return -1;
    }
    capture.open( -1 );
    if (!capture.isOpened())
    {
        printf("Error opening video capture\n"); return -1;
    }
    while (capture.read(frame))
    {
        if( frame.empty() )
        {
            printf("No captured frame -- Break!");
            break;
        }
        flip(frame, frame, 1);
        detectAndDisplay(frame, modelF, modelL);
        int c = waitKey(10);
        if( (char)c == 27 ) { break; }
    }
    return 0;
}

void detectAndDisplay( Mat frame, Ptr<FaceRecognizer> modelF, Ptr<LBPHFaceRecognizer> modelL)
{
    double t = 0, a = 0, g = 0;
    vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    t = (double)getTickCount();
    face_cascade.detectMultiScale(
        frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(150, 150)
    );
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle(frame, faces[i], CV_RGB(0, 255,0), 2);
        Mat faceROI = frame_gray( faces[i] );
        resize(faceROI, faceROI, Size(imageWidth, imageHeight));

        a = (double)getTickCount();
        int labelG = modelF->predict(faceROI);
        a = (double)getTickCount() - a;
        printf( "age prediction time = %g ms\n", a*1000/getTickFrequency());
        g = (double)getTickCount();
        int labelA = modelL->predict(faceROI);
        g = (double)getTickCount() - g;
        printf( "gender prediction time = %g ms\n", g*1000/getTickFrequency());
        string gender = "";
        if (labelG == 0)
        {
            gender = "Female";
        } else if (labelG == 1)
        {
            gender = "Male";
        }
        Point bottomleft( faces[i].x + faces[i].width/3, faces[i].y);
        putText(
            frame, gender + to_string(labelA), bottomleft, FONT_HERSHEY_PLAIN, 3, CV_RGB(0, 255, 0)
        );
    }
    imshow( window_name, frame );
}
