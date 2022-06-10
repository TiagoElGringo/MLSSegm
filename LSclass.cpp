#include "LSclass.h"
#include "LSSegm.h"

LSclass::LSclass() {

	par.mu = 0.005;
	par.a = 0;//600;
	par.sigma = 1.;
	par.eps = 1.5;
	par.kMax = 4;
	kcurr = par.kMax;
	par.nlevels = 5;
	edgeStopType = EDGE_STOP_INNER;


	if (edgeStopType == EDGE_STOP_INNER) {
		
	}
	else if (edgeStopType == EDGE_STOP_ALL) {
		par.aVec.resize(par.nlevels);
		for (int i = 0; i < par.aVec.size(); i++) {
			par.aVec[i] = 1;
		}
	}

	/*
	l = (float*)calloc(param.nlevels, sizeof(float));
	c = (float*)calloc(param.nlevels+1, sizeof(float));

	for (int i = 0; i < param.nlevels; i++) {
		l[i] = 10.;
	}
	*/

	itercurr = iter[kcurr];
	stepscurr = steps[kcurr];

	cout << "itercurr: " << itercurr << '\n';
	cout << "stepscurr: " << stepscurr << '\n';


	l.resize(par.nlevels);
	c.resize(par.nlevels + 1);

	//std::vector<float> l(par.nlevels, 0.0);
	//std::vector<float> c(par.nlevels+1, 0.0);

	for (int i = 0; i < l.size(); i++) {
		l[i] = i*10.;
	}

}

void LSclass::getImage(Mat image){
	timestart = high_resolution_clock::now();
	normalize(image, origImage, 0, 255, NORM_MINMAX);
	imagesc(origImage, "before"); //waitKey(0);
	truncateImage();
	imagesc(origImage, "after"); //waitKey(0);
	GaussianBlur(origImage, g0, Size(3,3), 0,0);
	resize(g0, g1, Size(), 0.5, 0.5, INTER_CUBIC); 
	resize(g1, g2, Size(), 0.5, 0.5, INTER_CUBIC); 
	resize(g2, g3, Size(), 0.5, 0.5, INTER_CUBIC); 
	resize(g3, g4, Size(), 0.5, 0.5, INTER_CUBIC);
	gcurr = g4;
	f = Mat::zeros(g4.size(), CV_32F);
}

void LSclass::standard() {

	drawMask();

	//cout << '\n' << f << "f" << '\n' << '\n';

	cv::FileStorage file("initialf.xml", cv::FileStorage::WRITE);
	file << "f" << (Mat1i)f;

	for (kcurr = par.kMax; kcurr >= 0; kcurr--) {
		cout << "k is now " << kcurr << '\n';
		itercurr = iter[kcurr];
		stepscurr = steps[kcurr];
		chanveseEvolve();
		//showResults();
		//cout << '\n' << '\n' << f << "f" << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		
		cv::FileStorage file(std::to_string(kcurr) + ".xml", cv::FileStorage::WRITE);
		file << "f" << (Mat1i)f;

		f.copyTo(fprev);
		if (kcurr != 0) {
			resize(fprev, f, Size(), 2, 2, INTER_LINEAR);
		}
		if (kcurr == par.kMax) {
			gcurr = g3;
		}
		else if (kcurr == par.kMax - 1) {
			gcurr = g2;
		}
		else if (kcurr == par.kMax - 2) {
			gcurr = g1;
		}
		else if (kcurr == par.kMax - 3) {
			gcurr = g0;
		}
	}
	timeend = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(timeend - timestart);
	cout << "Process time: " << duration.count() << endl;
	showResults();
	//cout << '\n' << '\n' << f << "f" << '\n' << '\n' << '\n' << '\n';
	//imagesc(f, "f"); waitKey(0);

}

void LSclass::chanveseEvolve() {
	float thismu = par.mu *255. * 255. * pow(2, -kcurr);

	//cout << f << '\n';

	for (int i = 0; i < stepscurr; i++) {
		//cout << "kcurr " << kcurr << " step number " << i << '\n';
		calc_c();
		//for (int i = 0; i < c.size(); i++) {
		//	cout << c[i] << " ";
		//}
		//cout << '\n';
		//for (int i = 0; i < itercurr; i++) {
			//nextIter(thismu);
			nextIterAlternative(thismu);
		//}
		//cout << '\n' << '\n' << f << "f" << '\n' << '\n' << '\n' << '\n'; waitKey(0);
	}
}

void LSclass::calc_c() {

	//cout << "sum c: " << sum(gcurr)[0] << '\n';
	//cout << "last element l: " << l.back() << '\n';

	c[0] = sum(gcurr.mul(hvi(l[0] - f)))[0] / sum(hvi(l[0] - f))[0];
	for (int i=1;i<l.size();i++){
		c[i] = sum(gcurr.mul(hvi(f - l[i - 1]).mul(hvi(l[i] - f))))[0] / sum(hvi(f - l[i - 1]).mul(hvi(l[i] - f)))[0];
	}
	c.back() = sum(gcurr.mul(hvi(f - l.back())))[0] / sum(hvi(f - l.back()))[0];
	/*
	for (int i = 0; i < c.size(); i++) {
		cout << c[i] << " ";
	}
	cout << '\n';
	
	waitKey(0);*/

}

void LSclass::nextIter(float mu) {

	C1 = calcRegTerm(1);
	C2 = calcRegTerm(2);
	C3 = calcRegTerm(3);
	C4 = calcRegTerm(4);
	/*
	cout << "C1: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	cout << "C2: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	cout << "C3: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	cout << "C4: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	*/

	diracTotal = cv::Mat::zeros(gcurr.size(), CV_32F);

	for (int i = 0; i < par.nlevels; i++) {
		diracTotal = diracTotal + dirac(f-l[i]);
	}

	cv::FileStorage file("diracTotal.xml", cv::FileStorage::WRITE);
	file << "f" << diracTotal;

	//cout << "diracTotal: " << diracTotal << '\n'; waitKey(0);

	for (int i = 0; i < itercurr; i++) {

		areaPart = dirac(l[0] - f).mul(square(gcurr - c[0]));
		for (int i = 1; i < par.nlevels; i++) {
			areaPart = areaPart + square(gcurr - c[i]).mul(dirac(l[i] - f).mul(hvi(f - l[i - 1])) - dirac(f - l[i - 1]).mul(hvi(l[i] - f)));
		}
		areaPart = areaPart - dirac(f - l.back()).mul(square(gcurr - c.back()));

		areaPart = areaPart.mul(diracTotal);

		m1 = mu * diracTotal;

		alpha = 1 + m1.mul(C1 + C2 + C3 + C4);

		calc_beta();

		calcEdgeStopTerm();

		f = (f + beta + areaPart - edgeStopPart).mul(1 / alpha);
		//f = (f + beta + areaPart).mul(1 / alpha);
		truncateLSF();

		//cout << areaPart << "areaPart: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		//cout << m1 << "m1: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		//cout << alpha << "alpha: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		//cout << beta << "beta: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		//cout << f << "f" << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);

	}
}

void LSclass::calcEdgeStopTerm() {
	
	/*Mat gradx, grady, result;
	
	Sobel(gcurr, gradx, -1, 1, 0, 1, .5, 0, BORDER_DEFAULT);
	Sobel(gcurr, grady, -1, 0, 1, 1, .5, 0, BORDER_DEFAULT);

	gradx = square(gradx);
	grady = square(grady);
	*/

	Mat x, y, kernelx, kernely;
	int depth = -1;
	Point anchorx, anchory;
	double delta = 0;

	float kernelElements1[] = { -1,1 };
	anchorx = cv::Point(0, 0);
	float kernelElements2[] = { -1,1 };
	anchory = cv::Point(0, 0);

	kernelx = cv::Mat(1, 2, CV_32F, kernelElements1);
	kernely = cv::Mat(2, 1, CV_32F, kernelElements2);

	cv::filter2D(gcurr, x, depth, kernelx, anchorx, delta, BORDER_DEFAULT);
	cv::filter2D(gcurr, y, depth, kernely, anchory, delta, BORDER_DEFAULT);

	x = square(x);
	y = square(y);

	edgeStopPart = edgestopFcn(x + y).mul(par.a);

	//edgeStopPart = inverseofgradient(gradx + grady).mul(par.a);

	edgeStopPart = edgeStopPart.mul(diracTotal);
}

void LSclass::calc_beta() {

	Mat f1, f2, f3, f4, kernel;
	int depth = -1;
	Point anchor;
	double delta = 0;

	float kElements1[] = { 0,1 };
	anchor = cv::Point(0, 0);
	kernel = cv::Mat(1, 2, CV_32F, kElements1);
	cv::filter2D(f, f1, depth, kernel, anchor, delta, BORDER_DEFAULT);

	float kElements2[] = { 1,0 };
	anchor = cv::Point(1, 0);
	kernel = cv::Mat(1, 2, CV_32F, kElements2);
	cv::filter2D(f, f2, depth, kernel, anchor, delta, BORDER_DEFAULT);

	float kElements3[] = { 0,1 };
	anchor = cv::Point(0, 0);
	kernel = cv::Mat(2, 1, CV_32F, kElements3);
	cv::filter2D(f, f3, depth, kernel, anchor, delta, BORDER_DEFAULT);

	float kElements4[] = { 1,0 };
	anchor = cv::Point(0, 1);
	kernel = cv::Mat(2, 1, CV_32F, kElements4);
	cv::filter2D(f, f4, depth, kernel, anchor, delta, BORDER_DEFAULT);

	beta = m1.mul(C1.mul(f1) + C2.mul(f2) + C3.mul(f3) + C4.mul(f4));
}

Mat LSclass::calcRegTerm(int dim) {

	Mat x, y, c, kernelx, kernely;
	int depth = -1;
	Point anchorx, anchory;
	double delta = 0;

	if (dim == 1 || dim == 3) {

		float kernelElements1[] = { -1,1 };
		anchorx = cv::Point(0, 0);
		float kernelElements2[] = { -1,1 };
		anchory = cv::Point(0, 0);

		kernelx = cv::Mat(1, 2, CV_32F, kernelElements1);
		kernely = cv::Mat(2, 1, CV_32F, kernelElements2);
	}
	else if (dim == 2) {
		float kernelElements1[] = { -1,1 };
		anchorx = cv::Point(1, 0);
		float kernelElements2[] = { -1,0,1,0 };
		anchory = cv::Point(1, 0);

		kernelx = cv::Mat(1, 2, CV_32F, kernelElements1);
		//cout << "kernelx" << kernelx << '\n';
		kernely = cv::Mat(2, 2, CV_32F, kernelElements2);
		//cout << "kernely"  << kernely << '\n';
	}
	else if (dim == 4) {
		float kernelElements1[] = { -1,1,0,0 };
		anchorx = cv::Point(0, 1);
		float kernelElements2[] = { -1,1 };
		anchory = cv::Point(0, 1);

		kernelx = cv::Mat(2, 2, CV_32F, kernelElements1);
		kernely = cv::Mat(2, 1, CV_32F, kernelElements2);
	}

	cv::filter2D(f, x, depth, kernelx, anchorx, delta, BORDER_DEFAULT);
	cv::filter2D(f, y, depth, kernely, anchory, delta, BORDER_DEFAULT);

	x = square(x);
	y = square(y);

	c = inverseofgradient(x + y);


	return c;
}

Mat LSclass::hvi(Mat f) {

	Mat fhvi;

	f.copyTo(fhvi);

	float eps = par.eps;

	//cout << "Before: " << f.at<float>(30, 20) << '\n';

	fhvi.forEach<float>([eps](float& p, const int* position) -> void {
		p = 0.5 + 1. / CV_PI * atan2(p, eps);
		});

	//cout << "After: " << fhvi.at<float>(30, 20) << '\n';

	return fhvi;
}

Mat LSclass::dirac(Mat f) {
	//Dirac = eps./(pi*(eps^2+(f).^2));
	//f.mul(f);

	//Mat image(576,768,CV_32F); 

	/*
	float p;

	for (int i = 0; i < f.rows; i++) {
		for (int j = 0; j < f.cols; j++) {
			f.at<float>(i, j) = param.eps / (CV_PI * (pow(param.eps, 2) + pow(f.at<float>(i, j),2)));
		}
	}
	*/

	//float* p;
	/*
	float eps = param.eps;

	struct Operator {
		float eps = LSclass::param.eps;
		void operator ()(float &p, const int* position) {
			p = param.eps / (CV_PI * (pow(eps, 2) + pow(p, 2)));
		}
	};
	image.forEach<Pixel>(Operator());
	*/

	Mat fdirac;

	f.copyTo(fdirac);

	float eps = par.eps;

	//cout << "Before: " << f.at<float>(30, 20) << '\n';

	fdirac.forEach<float>([eps](float &p, const int *position) -> void {
		p = eps / (CV_PI * (pow(eps, 2) + pow(p, 2)));
	});

	//cout << "After: " << fdirac.at<float>(30, 20) << '\n';

	return fdirac;

	/*
	image.forEach<Pixel>
	(
		[](Pixel& pixel, const int* position) -> void
		{
			pixel.x
		}
	);

	*/
	//forEach function (for actangents)

}

Mat LSclass::square(Mat g) {
	Mat result;

	g.copyTo(result);

	result.forEach<float>([](float& p, const int* position) -> void {
		p = pow(p,2);
		});

	return result;
}

Mat LSclass::inverseofgradient(Mat g) {
	Mat result;

	g.copyTo(result);

	float replacementVal = 0.0000001;

	result.forEach<float>([replacementVal](float& p, const int* position) -> void {
		if (p == 0){
			p = replacementVal;
		}
		p = 1/sqrt(p);
	});

	return result;
}

Mat LSclass::edgestopFcn(Mat g) {
	Mat result;

	g.copyTo(result);

	result.forEach<float>([](float& p, const int* position) -> void {
		p = 1 / (1+p);
		});

	return result;
}

void LSclass::drawMask(){

	const float R_singlemask = 15.;
	float R_circle = round(.3 * R_singlemask);

	Mat1b mask = cv::Mat::zeros(gcurr.size(),CV_32F);
	Mat1b invertedMask = cv::Mat::zeros(gcurr.size(), CV_32F);

	//cout << "mask type: " << mask.type() << '\n';

	for (int xc=R_singlemask; xc+R_circle<gcurr.cols; xc = xc + R_singlemask) { //while there is space to put more circles
		for (int yc=R_singlemask; yc + R_circle < gcurr.rows; yc = yc + R_singlemask) {
			//cout << xc << " " << yc << '\n';
			cv::circle(mask, Point(xc, yc), R_circle, Scalar(255), FILLED, LINE_8);
		}	
	} 

	//imshow("circles", (Mat1b)mask);
	//imshow("lsf zeros", (Mat1b)f);

	threshold(mask, mask, 1, 255, THRESH_BINARY);
	threshold(mask, invertedMask, 1, 255, THRESH_BINARY_INV);
/*
	imagesc(mask, "mask");
	imagesc(invertedMask, "invertedmask");

	imagesc(mask + invertedMask, "all ones");
*/
	Mat1f positivePart, negativePart;

	distanceTransform(mask, positivePart, DIST_L2, DIST_MASK_PRECISE);
	distanceTransform(invertedMask, negativePart, DIST_L2, DIST_MASK_PRECISE);
	negativePart = negativePart*(-1);


	f = positivePart + negativePart;

	//imagesc(f, "f before");

	Mat aux;

	threshold(f, aux, 0, 255, THRESH_BINARY);

	//imagesc(aux,"f > 0");

	//contour(f,0.);
	
}

void LSclass::imagesc(Mat image, String legend) {

	Mat normalized;

	normalize(image, normalized, 0, 255, NORM_MINMAX);
	imshow(legend, (Mat1b)normalized);

}

void LSclass::contour(Mat f, float level) {

	Mat aux;

	//f.copyTo(aux);

	threshold(f, aux, level, 255, THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	aux.convertTo(aux, CV_8U);

	//cout << "type of f: " << aux.type() << '\n';
	Scalar color;
	//waitKey(0);
	/*
	if (level == l[0]) {
		color = Scalar(255, 0, 0);
	}
	else if (level == l[1]) {
		color = Scalar(0, 255, 0);
	}
	else if (level == l[2]) {
		color = Scalar(0, 0, 255);
	}*/
	color = chooseColor(level);

	findContours(aux, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat cat_contours = Mat::zeros(aux.size(), CV_8UC3);

	for (size_t i = 0; i < contours.size(); i++)
		drawContours(cat_contours, contours, (int)i, color, 1, 8, hierarchy, 0, Point());

	//namedWindow(std::to_string(level), WINDOW_AUTOSIZE); imshow(std::to_string(level), cat_contours);
	
	cat_contours.copyTo(results, cat_contours); imshow("Results", results);
}

void LSclass::truncateImage() {
	
	origImage.forEach<float>([](float& p, const int* position) -> void {
		if (p >= 254) {
			p = 0;
		}
	});
	
};

void LSclass::showResults() {
	//namedWindow("results", WINDOW_AUTOSIZE); imshow("results", f);
	imshow("gcurr", gcurr);
	gcurr.copyTo(results); results = (Mat1b)results;
	//imshow("results", results);
	cvtColor(results,results,COLOR_GRAY2RGB);
	//imshow("resultsrgb", results);
	for (int i = 0; i < l.size(); i++) {
		contour(f,l[i]);
	}	
	//cout << '\n' << '\n' << f << "f" << '\n' << '\n';
	for (int i = 0; i < c.size();i++) {
		cout << c[i] << " ";
	}
	cout << '\n';

	// Declare what you need
	cv::FileStorage file("some_name.xml", cv::FileStorage::WRITE);

	// Write to file!
	file << "f" << (Mat1i)f;

	//waitKey(0);
}

Scalar LSclass::chooseColor(float level) {

	if (level == 0) {
		return Scalar(255, 0, 0); //blue
	}
	else if (level == 10) {
		return Scalar(0, 255, 0); //green
	}
	else if (level == 20) {
		return Scalar(0, 0, 255); //red
	}
	else if (level == 30) {
		return Scalar(0, 255, 255); //yellow
	}
	else if (level == 40) {
		return Scalar(0, 140, 255); //orange
	}
	else if (level == 50) {
		return Scalar(153, 0, 76);
	}
}

void LSclass::nextIterAlternative(float mu) {

	Mat diracTotal;

	C1 = calcRegTermAlternative(1);
	C2 = calcRegTermAlternative(2);
	alpha = Mat::zeros(f.size(), CV_32F);
	beta = Mat::zeros(f.size(), CV_32F);

	/*
	cout << "C1: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	cout << "C2: " << C1 << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; //waitKey(0);
	*/
	
	diracTotal = cv::Mat::zeros(gcurr.size(), CV_32F);

	for (int i = 0; i < par.nlevels; i++) {
		diracTotal = diracTotal + dirac(f - l[i]);
	}
	//cout << "diracTotal: " << diracTotal << '\n'; waitKey(0);

	m1 = mu * diracTotal;

	Mat aux1, aux2, aux3, aux4, auxf1, auxf2, auxf3, auxf4;

	aux1 = cv::Mat::zeros(gcurr.size(), CV_32F);
	aux2 = cv::Mat::zeros(gcurr.size(), CV_32F);
	aux3 = cv::Mat::zeros(gcurr.size(), CV_32F);
	aux4 = cv::Mat::zeros(gcurr.size(), CV_32F);
	auxf1 = cv::Mat::zeros(gcurr.size(), CV_32F);
	auxf2 = cv::Mat::zeros(gcurr.size(), CV_32F);
	auxf3 = cv::Mat::zeros(gcurr.size(), CV_32F);
	auxf4 = cv::Mat::zeros(gcurr.size(), CV_32F);

	C1.rowRange(Range(0, gcurr.rows - 2)).copyTo(aux1.rowRange(Range(1, gcurr.rows - 1)));
	C1.rowRange(Range(1, gcurr.rows - 1)).copyTo(aux2.rowRange(Range(1, gcurr.rows - 1)));
	C2.colRange(Range(0, gcurr.cols - 2)).copyTo(aux3.colRange(Range(1, gcurr.cols - 1)));
	C2.colRange(Range(1, gcurr.cols - 1)).copyTo(aux4.colRange(Range(1, gcurr.cols - 1)));

	alpha = 1 + m1.mul(aux1 + aux2 + aux3 + aux4);
	
	for (int i = 0; i < itercurr; i++) {
		
		areaPart = dirac(l[0] - f).mul(square(gcurr - c[0]));
		for (int i = 1; i < par.nlevels; i++) {
			areaPart = areaPart + square(gcurr - c[i]).mul(dirac(l[i] - f).mul(hvi(f - l[i - 1])) - dirac(f - l[i - 1]).mul(hvi(l[i] - f)));
		}
		areaPart = areaPart - dirac(f - l.back()).mul(square(gcurr - c.back()));

		areaPart = areaPart.mul(diracTotal);

		//cout << areaPart << "areaPart: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);

		f.rowRange(Range(0, gcurr.rows - 2)).copyTo(auxf1.rowRange(Range(1, gcurr.rows - 1)));
		f.rowRange(Range(2, gcurr.rows)).copyTo(auxf2.rowRange(Range(1, gcurr.rows - 1)));
		f.colRange(Range(0, gcurr.cols - 2)).copyTo(auxf3.colRange(Range(1, gcurr.cols - 1)));
		f.colRange(Range(2, gcurr.cols)).copyTo(auxf4.colRange(Range(1, gcurr.cols - 1)));

		beta = m1.mul(aux1.mul(auxf1) + aux2.mul(auxf2) + aux3.mul(auxf3) + aux4.mul(auxf4));

		/*
		int type;
		type = XDIFF;
		type = YDIFF;
		type = BDIFF;

		diracTotal = diff(C1, 1, -1);*/

		//cout << m1 << "m1: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		
		//alpha = 1 + m1.mul(C1 + C2 + diff(C1, XDIFF, BDIFF) + diff(C2, YDIFF, BDIFF));

		//cout << alpha << "alpha: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		
		//beta = diff(C1, XDIFF, BDIFF).mul(diff(f, XDIFF, BDIFF)) + C1.mul(diff(f, XDIFF, FDIFF)) + diff(C2, YDIFF, BDIFF).mul(diff(f, YDIFF, BDIFF)) + C2.mul(diff(f, YDIFF, FDIFF));

		//cout << beta << "beta: " << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);

		//Mat fant; f.copyTo(fant);

		f = (f + beta + areaPart).mul(1 / (alpha));

		truncateLSF();
		//f = f + areaPart;

		//cout << f << "f" << '\n' << '\n' << '\n' << '\n' << '\n' << '\n'; waitKey(0);
		
	}
	
}

Mat LSclass::calcRegTermAlternative(int dim) {

	Mat x, y, c, kernelx, kernely;
	int depth = -1;
	Point anchorx, anchory;
	double delta = 0;

	if (dim == 1) {

		float kernelElements1[] = { -1,1 };
		anchorx = cv::Point(1, 0);
		float kernelElements2[] = { -.5,0,.5 };
		anchory = cv::Point(0, 1);

		kernelx = cv::Mat(1, 2, CV_32F, kernelElements1);
		kernely = cv::Mat(3, 1, CV_32F, kernelElements2);
	}
	else if (dim == 2) {
		float kernelElements1[] = { -.5,0,.5 };
		anchorx = cv::Point(1, 0);
		float kernelElements2[] = { -1,1 };
		anchory = cv::Point(0, 1);

		kernelx = cv::Mat(1, 3, CV_32F, kernelElements1);
		//cout << "kernelx" << kernelx << '\n';
		kernely = cv::Mat(2, 1, CV_32F, kernelElements2);
		//cout << "kernely"  << kernely << '\n';
	}

	cv::filter2D(f, x, depth, kernelx, anchorx, delta, BORDER_DEFAULT);
	cv::filter2D(f, y, depth, kernely, anchory, delta, BORDER_DEFAULT);

	x = square(x);
	y = square(y);

	c = inverseofgradient(x + y);


	return c;
}

void LSclass::calcAlphaBetaAlternative() {
	
}

void LSclass::truncateLSF() {

	float truncateVal = -10;

	f.forEach<float>([truncateVal](float& p, const int* position) -> void {
	if (p < truncateVal) {
		p = truncateVal;
	}
	});
}

Mat LSclass::diff(Mat f, int dir, int type) {
	
	Mat result;
	Mat x, y, c, kernel;
	int depth = -1;
	Point anchor;
	double delta = 0;
	
	if (type == BDIFF)
		type = -1;
	else if (type == CDIFF)
		type = 0;
	else if (type == FDIFF)
		type = 1;

	if (dir == XDIFF){
		Sobel(f, result, -1, type, 0, 1, 1, 0, BORDER_DEFAULT);
	}
	else if (dir == YDIFF) {
		Sobel(f, result, -1, 0, type, 1, 1, 0, BORDER_DEFAULT);
	}

	cv::filter2D(f, result, depth, kernel, anchor, delta, BORDER_DEFAULT);
	

	return result;
}