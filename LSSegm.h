#ifndef LSSEGM_H
#define LSSEGM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <cstdio>
#include <chrono>

#define EDGE_STOP_NO 0
#define EDGE_STOP_INNER 1
#define EDGE_STOP_ALL 2

constexpr auto XDIFF = 1;
constexpr auto YDIFF = 2;
constexpr auto BDIFF = -1;
constexpr auto CDIFF = 0;
constexpr auto FDIFF = 1;

using namespace cv;
using namespace std;
using namespace std::chrono;



#endif
