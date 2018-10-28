#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include<dlib/opencv/cv_image_abstract.h>
//#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<vector>
#include<opencv2/photo/photo.hpp>
using namespace cv;
using namespace dlib;
using namespace std;


struct correspondens{
	std::vector<int> index;
};




void faceLandmarkDetection(dlib::array2d<unsigned char>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{
	dlib::frontal_face_detector detector = get_frontal_face_detector();
	//dlib::pyramid_up(img);

	std::vector<dlib::rectangle> dets = detector(img);
	


	full_object_detection shape = sp(img, dets[0]);
	//image_window win;
	//win.clear_overlay();
	//win.set_image(img);
	//win.add_overlay(render_face_detections(shape));
	for (int i = 0; i < shape.num_parts(); ++i)
	{
		float x = shape.part(i).x();
		float y = shape.part(i).y();
		landmark.push_back(Point2f(x, y));
	}


}






void delaunayTriangulation(const std::vector<Point2f>& hull, std::vector<correspondens>& delaunayTri, Rect rect)
{
	/////�ṹSubdiv2D�����˽����ڶ�ά�㼯�ϵĻ��ֽṹ�����е㼯���������ҹ���ƽ��ͼ�Σ�
	//��ͼ��ͨ�����һЩ���������ⲿ���ֵ㣨��Ϊ͹�ε㣩�ı�Ե����һ��ƽ���ð������Ե���ֳɺܶ�С����
	cv::Subdiv2D subdiv(rect);  ///ʹ���޶��ľ��δ���Subdiv2Dʵ��
	for (int it = 0; it < hull.size(); it++)
		subdiv.insert(hull[it]);   //insert ��ʾ��hull[it] ���β���subdiv ��

	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList); //��ȡdelaunay�ʷ������ε��б�
	
	//ʹ��subdiv.getVoronoiFacetList���Ի�ȡVoronoi����б�


	//std::vector<Point2f> pt;

	for (size_t i = 0; i < triangleList.size(); ++i)
	{

		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back(Point2f(t[0], t[1]));
		pt.push_back(Point2f(t[2], t[3]));
		pt.push_back(Point2f(t[4], t[5]));
		//cout<<"pt.size() is "<<pt.size()<<endl;

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<endl;
			int count = 0;
			for (int j = 0; j < 3; ++j)
			for (size_t k = 0; k < hull.size(); k++)
			if (abs(pt[j].x - hull[k].x) < 1.0   &&  abs(pt[j].y - hull[k].y) < 1.0)
			{
				ind.index.push_back(k);
				count++;
			}
			if (count == 3)
				//cout<<"index is "<<ind.index[0]<<" "<<ind.index[1]<<" "<<ind.index[2]<<endl;
				delaunayTri.push_back(ind);   ///�������ʷֵĽ���������
		}
		
	}


}




void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
	
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	
	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}






void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	
	std::vector<Point2f> t1Rect, t2Rect;
	std::vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); 

	}

	
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0); //�������亯���������ǵ㣬�ڽǵ���ɵĶ���������ָ������ɫ

	
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);  ////���з���任����

	multiply(img2Rect, mask, img2Rect);  ///����˷����൱��ֻ��ʾ���벻Ϊ0���ֵ�ͼ��
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));   ///Scalar(1.0, 1.0, 1.0) - mask�൱������ȡ��
	img2(r2) = img2(r2) + img2Rect;  ////��Ӻ��γ������ͱ����ֱ�Ϊ����ͼ��Ľ��

}










int main(int argc, char** argv)
{

	
	dlib::array2d<unsigned char> imgDlib1, imgDlib2;
	dlib::load_image(imgDlib1, "1.jpg");
	dlib::load_image(imgDlib2, "2.jpg");

	Mat imgCV1 = imread("1.jpg");
	Mat imgCV2 = imread("2.jpg");

	imshow("1", imgCV1);
	imshow("2", imgCV2);


	
	shape_predictor sp;
	deserialize("D:\\Dlib\\shape_predictor_68_face_landmarks.dat") >> sp;
	std::vector<Point2f> points1, points2;

	faceLandmarkDetection(imgDlib1, sp, points1);
	faceLandmarkDetection(imgDlib2, sp, points2);



	
	Mat imgCV1Warped = imgCV2.clone();
	imgCV1.convertTo(imgCV1, CV_32F);
	imgCV1Warped.convertTo(imgCV1Warped, CV_32F);

	std::vector<Point2f> hull1;
	std::vector<Point2f> hull2;
	std::vector<int> hullIndex;

	cv::convexHull(points2, hullIndex, false, false);  //����͹����ѡ������2��ͼ���͹���㣬��Ϊ2��ͼ��������ĵװ�ͼ��

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);  ////1��ͼ���ȡ�ĵ�����Ҫ���������͹���ĵ�����һ��
		hull2.push_back(points2[hullIndex[i]]);
	}



	std::vector<correspondens> delaunayTri;
	Rect rect(0, 0, imgCV1Warped.cols, imgCV1Warped.rows); ///�ʷ�ǰѡ��һ��������������Ҫ�ʷֵ����е�
	delaunayTriangulation(hull2, delaunayTri, rect);  ////���������ʷ�  �� һ�ֶ�ά������ά�ṹ�ķ���

	for (size_t i = 0; i<delaunayTri.size(); ++i)
	{
		std::vector<Point2f> t1, t2;
		correspondens corpd = delaunayTri[i];
		for (size_t j = 0; j<3; ++j)
		{
			t1.push_back(hull1[corpd.index[j]]);
			t2.push_back(hull2[corpd.index[j]]);
		}

		warpTriangle(imgCV1, imgCV1Warped, t1, t2);/////����һһ��Ӧ�������ʷֽ����������ͼ��һһ��Ӧ�Ĳ���ͨ��forѭ�������滻
	}


	
	std::vector<Point> hull8U;

	for (int i = 0; i< hull2.size(); ++i)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}


	Mat mask = Mat::zeros(imgCV2.rows, imgCV2.cols, imgCV2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255)); //�������亯���������ǵ㣬�ڽǵ���ɵĶ���������ָ������ɫ


	Rect r = boundingRect(hull2);
	Point center = (r.tl() + r.br()) / 2;

	Mat output;
	imgCV1Warped.convertTo(imgCV1Warped, CV_8UC3);
	seamlessClone(imgCV1Warped, imgCV2, mask, center, output, NORMAL_CLONE); //�����ںϣ�Ϊ����ƴ�ӵı�Ե�������
	
	//string filename = argv[1];
	//filename = filename + argv[2];
	//filename = filename + ".jpg";
	//imwrite(filename, output);
	imshow("����", output);
	waitKey(0);
	//destroyAllWindows();

	return 0;
}
