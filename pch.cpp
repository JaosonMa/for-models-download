
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Windows.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace cv;
using namespace cv::dnn;
using namespace std;
string labels_txt_file = R"(F:\self_data\classification\models\infer\labels.txt)";
string tf_pb_file = R"(F:/self_data/classification/models/infer/frozen_graph.pb)";
vector <string> readClassNames();
void main()
{
	Net net = readNetFromTensorflow(tf_pb_file);
	Mat src = imread(R"(F:\self_data\classification\val_dir\N\2_1_25_3_1.jpg)");
	if (src.empty()) {
		cout << "error:no img" << endl;
	}
	vector <string> labels = readClassNames();
	int w = 224;
	int h = 224;
	resize(src, src, Size(w, h));

	DWORD timestart = GetTickCount();
	if (net.empty()) {
		cout << "error:no model" << endl;
	}
	Mat inputBlob = blobFromImage(src, 0.00390625f, Size(w, h), Scalar(), true, false);
	Mat prob;
	net.setInput(inputBlob, "ExpandDims");
	prob = net.forward("MobilenetV2/Predictions/Reshape_1");
	cout << prob << endl;
	//prob=net.forward("softmax2");
	//得到最大分类概率
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	DWORD timeend = GetTickCount();
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
	cout << "用时(毫秒):" << timeend - timestart << endl;
	// 显示文本
	putText(src, labels.at(classidx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("Image Classfication", src);
	waitKey(0);

}
vector <string>readClassNames()
{
	vector <string>classNames;
	fstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		cout << "does not open" << endl;
		exit(-1);
	}
	string name;
	while (!fp.eof())
	{
		getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}


