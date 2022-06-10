// LSSegm.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.
//

/*#include <iostream>

int main()
{
    std::cout << "Hello World!\n";
}
*/
// Executar programa: Ctrl + F5 ou Menu Depurar > Iniciar Sem Depuração
// Depurar programa: F5 ou menu Depurar > Iniciar Depuração

// Dicas para Começar: 
//   1. Use a janela do Gerenciador de Soluções para adicionar/gerenciar arquivos
//   2. Use a janela do Team Explorer para conectar-se ao controle do código-fonte
//   3. Use a janela de Saída para ver mensagens de saída do build e outras mensagens
//   4. Use a janela Lista de Erros para exibir erros
//   5. Ir Para o Projeto > Adicionar Novo Item para criar novos arquivos de código, ou Projeto > Adicionar Item Existente para adicionar arquivos de código existentes ao projeto
//   6. No futuro, para abrir este projeto novamente, vá para Arquivo > Abrir > Projeto e selecione o arquivo. sln

#include "LSSegm.h"
#include "LSclass.h"

using namespace cv;

int main(int argc, char** argv)
{
	std::ios::sync_with_stdio(false);

	Mat3f image;
	
	Mat1f gray_image, blurredImage;

	image = imread(argv[1], IMREAD_COLOR);

	if (argc != 2 || !image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	char* imageName = argv[1];

	cvtColor(image, gray_image, COLOR_RGB2GRAY); 
	//GaussianBlur(gray_image, blurredImage, Size(5,5), 0, 0);

	//imwrite("../../images/Gray_Image.jpg", gray_image);

	//namedWindow(imageName, WINDOW_AUTOSIZE); imshow(imageName, image);
	//namedWindow("Gray image", WINDOW_AUTOSIZE); imshow("Gray image", gray_image);
	//namedWindow("Blurred image", WINDOW_AUTOSIZE); imshow("Blurred image", (Mat1b)blurredImage);

	LSclass data;
	auto start = high_resolution_clock::now();
	data.getImage(gray_image);
	/*
	//Mat gradx, grady, x, y;
	Mat test2, test3;
	test2 = Mat::zeros(4, 5, CV_32F);
	test3 = Mat::zeros(4, 5, CV_32F);
	float testElements[] = { 1,5,2,4,10,6,7,3,2,8,4,6,0,2,9,11,13,2,1,8};
	Mat test = cv::Mat(4, 5, CV_32F, testElements);

	test2(Range(0, 2), Range(0, 5)) = test(Range(1, 3), Range(0, 5));

	test(Range(1, 3), Range(0, 5)).copyTo(test2(Range(0, 2), Range(0, 5)));
	test(Range(1, 3), Range(0, 5)).copyTo(test3);


	cout << test << "test " << '\n';
	cout << test(Range(1, 3), Range(0, 5)) << '\n';
	cout << test2 << '\n';
	cout << test3 << '\n';*/
	

	/*

	Sobel(test, gradx, -1, 1, 0, 1, 1, 0, BORDER_DEFAULT);
	Sobel(test, grady, -1, 0, 1, 1, 1, 0, BORDER_DEFAULT);
	
	float kernelElements1[] = {0,-1,1};
	Point anchorx = cv::Point(1, 0);
	float kernelElements2[] = { -.5,0,.5 };
	Point anchory = cv::Point(0, 1);

	int depth = -1;
	double delta = 0;

	Mat kernelx = cv::Mat(1, 2, CV_32F, kernelElements1);
	Mat kernely = cv::Mat(3, 1, CV_32F, kernelElements2);

	cv::filter2D(test, x, depth, kernelx, anchorx, delta, BORDER_DEFAULT);
	cv::filter2D(test, y, depth, kernely, anchory, delta, BORDER_DEFAULT);

	cout << test << "test " << '\n';
	cout << gradx << "gradx " << '\n';
	cout << grady << "grady " << '\n';
	cout << x << " x " << '\n';
	cout << y << " y" << '\n';*/
	
	namedWindow("Results", WINDOW_AUTOSIZE);



	//data.drawMask();
	data.standard();

	//namedWindow("pocrl", WINDOW_AUTOSIZE); imshow("pocrl", data.LSF);
	//namedWindow("origfloat", WINDOW_AUTOSIZE); imshow("origfloat", (Mat1b)imagefloat);
	//imshow("origfloat", reducedImage);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Total time: " << duration.count() << endl;
	waitKey(0);

	return 0;
}