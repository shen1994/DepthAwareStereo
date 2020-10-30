#include <iostream>
#include "dcnnmodule.h"

int main(int argc, char **argv) {

    // initialize
    DCNNModule dModel = DCNNModule("../models/dcnn20190902.pt", "../models/config20190902.ini", 0);
    dModel.DCNN_initialize();
     
    // read and deal
    cv::Mat image = cv::imread("/home/westwell/Desktop/DCNNDemo/src/image_test.png", 1);
    cv::Mat depth = cv::imread("/home/westwell/Desktop/DCNNDemo/src/depth_test.png", 0);

    // define timer
    double time_used = 0.;
    struct timeval time_start,time_stop;
    
    cv::Mat segout = cv::Mat(400, 700, CV_8UC1);
    for(int i = 0;i < 100; i++){
    //while(true){  
      gettimeofday(&time_start, nullptr);
      
      dModel.DCNN_core_compute(image, depth, segout);
      
      gettimeofday(&time_stop, nullptr);
      time_used += 1000000*(time_stop.tv_sec-time_start.tv_sec)+time_stop.tv_usec-time_start.tv_usec;
          
      // write one image to somewhere
      cv::imwrite("/home/westwell/Desktop/DCNNDemo/src/segment.png", segout);
      std::cout<<"westwell: Image has been wroten to file."<<std::endl;
    }
    segout.release();
    
    // print time used
    std::cout<<time_used / 1000. / 100.<<std::endl;
     
    // release all resources
    dModel.DCNN_release();

    return 0;
}
