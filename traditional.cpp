#include <iostream>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/video.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <random>

const int FMS_BKGND = 50;
int minArea = 5;
int thresholdValue = 30;

static void on_tracker(int pos, void* data) {
  minArea = pos;
}



cv::Mat getBackground(cv::Mat* images_ptr, int size = FMS_BKGND) {
  std::cout << "Getting background...";
  int rows = images_ptr[0].rows;
  int cols = images_ptr[0].cols;
  
  cv::Mat background(rows, cols, CV_8UC1);  //store the background image
  for (int x = 0; x < cols; x++) {
    for (int y = 0; y < rows; y++) {
      //create an array with the (x,y) pixel of all frames
      uchar currentPixel[size];
      for (int i = 0; i < size; i++) {
        //insert sort: pos is the position where the element will be inserted
        int pos = i-1;
        while(pos>=0 && images_ptr[i].at<uchar>(y,x) < currentPixel[pos]) {
          currentPixel[pos+1] = currentPixel[pos];
          pos--;
        }
        currentPixel[pos+1] = images_ptr[i].at<uchar>(y,x);
      }
      //now currentPixel is a sorted array with (x,y) pixel by all frames.
      //gets the median value and write it to the (x,y) pixel in background image
      background.at<uchar>(y,x) = currentPixel[size/2];
    }
  }
  std::cout << "     Done\n";
  return background;
}
//returns the background from a video
cv::Mat getBackground(std::string path) {
  std::cout << "Getting background from video " << path << "...\n";
  //TO DO: check if the file has a video extension
  cv::VideoCapture cap(path);
  //take 50 random frames
  cv::Mat frames[FMS_BKGND];
  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, cap.get(cv::CAP_PROP_FRAME_COUNT)-1);
  for (int i = 0; i < FMS_BKGND; i++) {
    int rand = dist(rd);
    cap.set(cv::CAP_PROP_POS_FRAMES, rand); //set the frame id to read that particular frame
    cap.read(frames[i]);  //read that frame
    cv::cvtColor(frames[i], frames[i], cv::COLOR_BGR2GRAY); //convert it in B&W
  }
  cap.release();

  //calculate the background with those frames
  return getBackground(frames);
}


double calculateFPS(cv::VideoCapture& cap) {
  double fps = cap.get(cv::CAP_PROP_FPS);
  
  if (fps != 0)
    return fps;

  //if the property is zero, try to calculate it in a different way
  int num_frames = 60;
  time_t start, end;
  cv::Mat frame;

  time(&start);
  for (int i = 0; i < num_frames; i++) {
    cap >> frame;
  }
  time(&end);

  double seconds = difftime(end, start);
  fps = num_frames / seconds;
  return fps;
}


int main(int argc, char *argv[]) {
  std::string filename;
  int delay = 30;
  bool showVideo = true;
  int BGchoice = 0; //0 = no, 1 = calculate, 2 = load

  //take filename from command line
  if (argc == 5) {
    filename = argv[1];
    delay = std::stoi(argv[2]);
    showVideo = std::stoi(argv[3]);
    BGchoice = std::stoi(argv[4]);
  }
  else if (argc == 2) {
    filename = argv[1];
  }
  else {
    std::cerr << "Usage: " << argv[0] << " <filename> [delay] [showVideo] [calculateBG]\n";
    return -1;
  }

  // std::cout << "Filename: " << filename << "\n";
  // std::cout << "Delay: " << delay << "\n";
  // std::cout << "Show video: " << showVideo << "\n";
  // std::cout << "Calculate background: " << calculateBG << "\n";


  cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
  // cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorKNN();
  cv::Mat frame, fgMask, bg;

  if (BGchoice == 1) {
    std::cout << "Calculating background...\n";
    bg = getBackground(filename);
    cv::imwrite("background.jpg", bg);
  }
  else if (BGchoice == 2) {
    std::cout << "Loading background...\n";
    bg = cv::imread("background.jpg", cv::IMREAD_GRAYSCALE);
  }
  else {
    std::cout << "No background loaded\n";
  }

  std::cout << "Done BG\n";

  std::string winName = "Trackbars";
  cv::namedWindow(winName);
  cv::createTrackbar("Min Area", winName, &minArea, 100, on_tracker);
  cv::createTrackbar("Threshold", winName, &thresholdValue, 255);

  cv::VideoCapture cap(filename);

  if (!cap.isOpened()) {
    std::cerr << "Error! Unable to open video file " << filename << ".\n";
    return -1;
  }

  double fps = calculateFPS(cap);
  std::cout << "Frames per second: " << fps << "\n";

  //process

  cap >> frame;
  if (frame.empty()) {
    std::cerr << "Error! Unable to read first frame.\n";
    return -1;
  }
  cv::VideoWriter wtr(filename + "-output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame.cols, frame.rows), true);

  while (true) {
    cap >> frame;
    if (frame.empty())
      break;

    if (BGchoice == 0) {
      pBackSub->apply(frame, fgMask);
      cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);   //remove shadows
    }
    else {
      fgMask = cv::Mat(frame.size(), CV_32FC1);   //gray frame
      cv::cvtColor(frame, fgMask, cv::COLOR_BGR2GRAY);
      cv::absdiff(fgMask, bg, fgMask);
      cv::threshold(fgMask, fgMask, thresholdValue, 255, cv::THRESH_BINARY);
    }
    
    cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1, -1), 1);
    cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(frame, contours, -1, cv::Scalar(255, 0, 0), 2);

    // for (auto contour : contours) {
    //   if (cv::contourArea(contour) < minArea)
    //     continue;

    //   cv::Rect r = cv::boundingRect(contour);
    //   cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
    // }

    wtr << frame;

    if (showVideo) {
      cv::imshow("Frame", frame);
      cv::imshow("FG Mask", fgMask);

      if (cv::waitKey(delay) == 'q')
        break;
    }
  }

  return 0;
}