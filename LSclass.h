#include "LSsegm.h"

typedef struct LSparam {
	float mu;			//length regularization term
	vector<float> aVec;	//weight of stop-by-gradient term	
	float a;			//(inner contour) stopping-by-gradient term
	float eps;			//heviside and dirac fcns regularization term
	float sigma;		//smoothing term
	float dt;			//time step

	int nlevels;

	int kMax;			//initial level of coarser image
}LSparam;


class LSclass{

	public:
		LSclass();

		LSparam par;
		Mat origImage;
		Mat gcurr;
		Mat f;
		Mat fprev;

		Mat g0; 
		Mat g1;
		Mat g2;
		Mat g3;
		Mat g4;

		Mat results;

		Mat diracTotal, C1, C2, C3, C4, m1, alpha, beta, areaPart, edgeStopPart;

		int edgeStopType;

		int kcurr;
		int stepscurr;
		int itercurr;
		high_resolution_clock::time_point timestart,timeend;


		vector<float> c;
		vector<float> l;
		vector<int> steps{1,2,20,20,20};
		vector<int> iter{1,1,20,20,20};

		//void drawMask();

		void getImage(Mat image);

		//void create_multiple_images();
		
		Mat hvi(Mat f);
		Mat dirac(Mat f);
		//Mat variation(Mat g, float c);
		Mat square(Mat f);
		void calc_c();
		void nextIter(float thismu);
		void chanveseEvolve();
		Mat calcRegTerm(int dim);

		Mat inverseofgradient(Mat g);
		Mat edgestopFcn(Mat g);

		void drawMask();
		void imagesc(Mat image, String legend);
		void standard();
		void contour(Mat f, float level);
		Mat greaterThan(Mat f, float l);
		void calc_beta();
		void truncateImage();
		void showResults();
		void calcEdgeStopTerm();

		void calcAlphaBetaAlternative();
		void nextIterAlternative(float thismu);
		Mat calcRegTermAlternative(int dim);
		void truncateLSF();

		Mat diff(Mat f,int dir, int type);

		Scalar chooseColor(float level);



};

