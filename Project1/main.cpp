//#include"onnx.h"
//#include "fov_puzzle.h"
//#include<opencv2/opencv.hpp>
//#include<time.h>
//
//using namespace cv;
//using namespace std;
////
//int main()
//{
//    //测试fov拼图
//    fov_scan fov;
//    vector<Rect2f> fov1;
//    vector<Mat> temp;
//    Mat a = imread("C:\\Users\\LENOVO\\Pictures\\1\\1.jpg");
//    Mat b = imread("C:\\Users\\LENOVO\\Pictures\\1\\2.jpg");
//    Mat c = imread("C:\\Users\\LENOVO\\Pictures\\1\\3.jpg");
//    Mat d = imread("C:\\Users\\LENOVO\\Pictures\\1\\4.jpg");
//    Mat e = imread("C:\\Users\\LENOVO\\Pictures\\1\\5.jpg");
//    Mat f = imread("C:\\Users\\LENOVO\\Pictures\\1\\6.jpg");
//    Mat g = imread("C:\\Users\\LENOVO\\Pictures\\1\\7.jpg");
//    Mat h = imread("C:\\Users\\LENOVO\\Pictures\\1\\8.jpg");
//    Mat i = imread("C:\\Users\\LENOVO\\Pictures\\1\\9.jpg");
//    Mat j = imread("C:\\Users\\LENOVO\\Pictures\\1\\10.jpg");
//    Mat k = imread("C:\\Users\\LENOVO\\Pictures\\1\\11.jpg");
//    Mat l = imread("C:\\Users\\LENOVO\\Pictures\\1\\12.jpg");
//    Mat m = imread("C:\\Users\\LENOVO\\Pictures\\1\\13.jpg");
//    Mat n = imread("C:\\Users\\LENOVO\\Pictures\\1\\14.jpg");
//    Mat o = imread("C:\\Users\\LENOVO\\Pictures\\1\\15.jpg");
//    Mat p = imread("C:\\Users\\LENOVO\\Pictures\\1\\16.jpg");
//    Mat q = imread("C:\\Users\\LENOVO\\Pictures\\1\\17.jpg");
//    Mat r = imread("C:\\Users\\LENOVO\\Pictures\\1\\18.jpg");
//    Mat s = imread("C:\\Users\\LENOVO\\Pictures\\1\\19.jpg");
//    Mat t = imread("C:\\Users\\LENOVO\\Pictures\\1\\20.jpg");
//    Mat u = imread("C:\\Users\\LENOVO\\Pictures\\1\\21.jpg");
//    Mat v = imread("C:\\Users\\LENOVO\\Pictures\\1\\22.jpg");
//    Mat w = imread("C:\\Users\\LENOVO\\Pictures\\1\\23.jpg");
//    Mat x = imread("C:\\Users\\LENOVO\\Pictures\\1\\24.jpg");
//    Mat y = imread("C:\\Users\\LENOVO\\Pictures\\1\\25.jpg");
//    Mat aa = imread("C:\\Users\\LENOVO\\Pictures\\1\\26.jpg");
//    Mat bb = imread("C:\\Users\\LENOVO\\Pictures\\1\\27.jpg");
//    Mat cc = imread("C:\\Users\\LENOVO\\Pictures\\1\\28.jpg");
//    Mat dd = imread("C:\\Users\\LENOVO\\Pictures\\1\\29.jpg");
//    Mat ee = imread("C:\\Users\\LENOVO\\Pictures\\1\\30.jpg");
//    Mat ff = imread("C:\\Users\\LENOVO\\Pictures\\1\\31.jpg");
//    Mat gg = imread("C:\\Users\\LENOVO\\Pictures\\1\\32.jpg");
//    Mat hh = imread("C:\\Users\\LENOVO\\Pictures\\1\\33.jpg");
//    Mat ii = imread("C:\\Users\\LENOVO\\Pictures\\1\\34.jpg");
//    Mat jj = imread("C:\\Users\\LENOVO\\Pictures\\1\\35.jpg");
//    Mat kk = imread("C:\\Users\\LENOVO\\Pictures\\1\\36.jpg");
//    Mat ll = imread("C:\\Users\\LENOVO\\Pictures\\1\\37.jpg");
//    Mat mm = imread("C:\\Users\\LENOVO\\Pictures\\1\\38.jpg");
//    Mat nn = imread("C:\\Users\\LENOVO\\Pictures\\1\\39.jpg");
//    temp.push_back(a);
//    temp.push_back(b);
//    temp.push_back(c);
//    temp.push_back(d);
//    temp.push_back(e);
//    temp.push_back(f);
//    temp.push_back(g);
//    temp.push_back(h);
//    temp.push_back(i);
//    temp.push_back(j);
//    temp.push_back(k);
//    temp.push_back(l);
//    temp.push_back(m);
//    temp.push_back(n);
//    temp.push_back(o);
//    temp.push_back(p);
//    temp.push_back(q);
//    temp.push_back(r);
//    temp.push_back(s);
//    temp.push_back(t);
//    temp.push_back(u);
//    temp.push_back(v);
//    temp.push_back(w);
//    temp.push_back(x);
//    temp.push_back(y);
//    temp.push_back(aa);
//    temp.push_back(bb);
//    temp.push_back(cc);
//    temp.push_back(dd);
//    temp.push_back(ee);
//    temp.push_back(ff);
//    temp.push_back(gg);
//    temp.push_back(hh);
//    temp.push_back(ii);
//    temp.push_back(jj);
//    temp.push_back(kk);
//    temp.push_back(ll);
//    temp.push_back(mm);
//    temp.push_back(nn);
//
//    Rect2f aaa(-108, 3324, 2448, 2048);
//    Rect2f bbb(17224, 10092, 2448, 2048);
//    Rect2f ccc(15579, 7482.5, 2448, 2048);
//    Rect2f ddd(15074.5, 6381.5, 2448, 2048);
//    Rect2f eee(16921.5, 7097, 2448, 2048);
//    Rect2f fff(16729.5, 5680.5, 2448, 2048);
//    Rect2f zzz(16680.5 , 5169.5, 2448, 2048);
//    Rect2f ggg(17051, 2573, 2448, 2048);
//    Rect2f hhh(15985, 3640.5, 2448, 2048);
//    Rect2f iii(13518, 3518.5, 2448, 2048);
//    Rect2f jjj(11563, 2632, 2448, 2048);
//    Rect2f kkk(10160.5, 3830.5, 2448, 2048);
//    Rect2f lll(7971, 3432.5, 2448, 2048);
//    Rect2f mmm(6470, 2800.5, 2448, 2048);
//    Rect2f nnn(4237, 2910.5, 2448, 2048);
//    Rect2f ooo(2009.5, 3284.5, 2448, 2048);
//    Rect2f ppp(463.5, 2800.5, 2448, 2048);
//    Rect2f qqq(897.5, 4733.5, 2448, 2048);
//    Rect2f rrr(776.5, 6177, 2448, 2048);
//    Rect2f sss(1253.5, 7219, 2448, 2048);
//    Rect2f ttt(1247, 9178, 2448, 2048);
//    Rect2f uuu(3619, 9500, 2448, 2048);
//    Rect2f vvv(3613.5, 7500.5, 2448, 2048);
//    Rect2f www(3042.5, 6399.5, 2448, 2048);
//    Rect2f xxx(5430, 4736.5, 2448, 2048);
//    Rect2f yyy(5648.5, 5569, 2448, 2048);
//    Rect2f aaaa(7158.5, 5127.5,2448,2048);
//    Rect2f bbbb(8165, 5576.5,2448,2048);
//    Rect2f cccc(5824.5, 7153.5,2448,2048);
//    Rect2f dddd(7274, 9156.5,2448,2048);
//    Rect2f eeee(9602, 9592,2448,2048);
//    Rect2f ffff(8384.5, 7478,2448,2048);
//    Rect2f gggg(10354.5, 7133.5,2448,2048);
//    Rect2f hhhh(10286, 5651,2448,2048);
//    Rect2f iiii(11658, 6105,2448,2048);
//    Rect2f jjjj(13243, 5376.5,2448,2048);
//    Rect2f kkkk(13222, 7180.5,2448,2048);
//    Rect2f llll(13236.5, 9141.5,2448,2048);
//    Rect2f mmmm(15597.5, 10064,2448,2048);//1010
//    /*Rect2f eee(1000, 1000, 200, 200);
//    Rect2f fff(1000, 1500, 200, 200);*/
//    fov1.push_back(aaa);
//    fov1.push_back(bbb);
//    fov1.push_back(ccc);
//    fov1.push_back(ddd);
//    fov1.push_back(eee);
//    fov1.push_back(fff);
//    fov1.push_back(zzz);
//    fov1.push_back(ggg);
//    fov1.push_back(hhh);
//    fov1.push_back(iii);
//    fov1.push_back(jjj);
//    fov1.push_back(kkk);
//    fov1.push_back(lll);
//    fov1.push_back(mmm);
//    fov1.push_back(nnn);
//    fov1.push_back(ooo);
//    fov1.push_back(ppp);
//    fov1.push_back(qqq);
//    fov1.push_back(rrr);
//    fov1.push_back(sss);
//    fov1.push_back(ttt);
//    fov1.push_back(uuu);
//    fov1.push_back(vvv);
//    fov1.push_back(www);
//    fov1.push_back(xxx);
//    fov1.push_back(yyy);
//    fov1.push_back(aaaa);
//    fov1.push_back(bbbb);
//    fov1.push_back(cccc);
//    fov1.push_back(dddd);
//    fov1.push_back(eeee);
//    fov1.push_back(ffff);
//    fov1.push_back(gggg);
//    fov1.push_back(hhhh);
//    fov1.push_back(iiii);
//    fov1.push_back(jjjj);
//    fov1.push_back(kkkk);
//    fov1.push_back(llll);
//    fov1.push_back(mmmm);
//    
//    /*  fov1.push_back(eee);
//      fov1.push_back(fff);*/
//
//    Mat lalla = fov.fov_puzzle(fov1, temp,);
//    namedWindow("1", 0);
//    imshow("1", lalla);
//    waitKey(0);
//    return 0;
//}



//    //测试贴片一键搜索
//    clock_t start, endd;
//    Net_config yolo_nets = { 0.4, 0.4, 0.4,"best17.onnx" };//best_rpc.onnx   10.12_rpc
//    YOLO yolo_model(yolo_nets);
//
//    cv::String pattern = "C:\\Users\\LENOVO\\Pictures\\test_rpc.jpg"; /*C:\\Users\\LENOVO\\Pictures\\222.jpg*/
//    vector<string>img_path;
//    vector<cv::String>fn;
//    glob(pattern, fn, false);
//    size_t count = fn.size();
//    for (size_t j = 0; j < count; j++)
//    {
//        img_path.push_back(fn[j]);
//        cout << "!" << fn[j] << endl;
//    }
//    vector<cv::Rect2f> smt_frame;
//    for (int i = 0; i < img_path.size(); i++)
//    {
//        string a = to_string(i + 1);// 1开始命名存储
//        Mat srcimg = cv::imread(img_path[i]);
//        start = clock();//开始计时
//        smt_frame = yolo_model.detect(srcimg);
//        cv::imwrite(a + "10_31_2dst_rpc.jpg", srcimg);
//        endd = clock();//结束计时
//        double endtime = (double)(endd - start);
//        cout << "Total time:" << endtime << "ms" << endl;	//ms为单位
//
//    }
//    for (int i = 0; i < smt_frame.size(); i++)
//    {
//        cout << "!!" << smt_frame[i].x << "!!" << smt_frame[i].y  << endl;
//    }
//    return 0;
//}




    //测试缺陷检测智能方法
//#include <fstream>
//#include <sstream>
//#include <iostream>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
////#include <cuda_provider_factory.h>
//#include <onnxruntime_cxx_api.h>
//
//using namespace std;
//using namespace cv;
//using namespace Ort;
//
///****************************************
//@brief     :  分割onnxruntime
//@input	   :  图像
//@output    :  掩膜
//*****************************************/
//
//void center_resize(cv::Mat& src)//中心裁剪方法2
//{
//	int width = src.cols;
//	int height = src.rows;
//	Point center(width / 2, height / 2); // 指定的中心点 
//	Size size(300, 300); // 裁剪出的图片大小 
//	getRectSubPix(src, size, center, src); // 裁剪出目标图片
//}
//void SegmentAIONNX(Mat& imgSrc, int width, int height)
//{
//	//模型信息 /
//	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
//	Ort::SessionOptions session_options;
//	session_options.SetIntraOpNumThreads(1);
//
//	const wchar_t* model_path = L"acc-76.onnx";
//
//	Ort::Session session(env, model_path, session_options);
//	Ort::AllocatorWithDefaultOptions allocator;
//	size_t num_input_nodes = session.GetInputCount();	//batchsize
//	size_t num_output_nodes = session.GetOutputCount();
//	const char* input_name = session.GetInputName(0, allocator);
//	const char* output_name = session.GetOutputName(0, allocator);
//	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();	//输入输出维度
//	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
//	std::vector<const char*> input_names{ input_name };
//	std::vector<const char*> output_names = { output_name };
//	clock_t startTime, endTime;	//计算推理时间
//	//startTime = clock();
//	//输入处理//
//	Mat imgBGR = imgSrc;	//输入图片预处理
//	Mat imgRGBresize;
//	center_resize(imgBGR);
//	//resize(imgBGR, imgBGRresize, Size(input_dims[3], input_dims[2]), InterpolationFlags::INTER_CUBIC);
//
//	cvtColor(imgBGR, imgRGBresize, COLOR_BGR2RGB);	//smp未转RGB
//	Mat resize_img;
//	imgRGBresize.convertTo(resize_img, CV_32F, 1.0 / 255);  //divided by 255转float
//	cv::Mat channels[3]; //分离通道进行HWC->CHW
//	cv::split(resize_img, channels);
//	std::vector<float> inputTensorValues;
//	float mean[] = { 0.485f, 0.456f, 0.406f };	//
//	float std_val[] = { 0.229f, 0.224f, 0.225f };
//	for (int i = 0; i < resize_img.channels(); i++)	//标准化ImageNet
//	{
//		channels[i] -= mean[i];  // mean均值
//		channels[i] /= std_val[i];   // std方差
//	}
//	for (int i = 0; i < resize_img.channels(); i++)  //HWC->CHW
//	{
//		std::vector<float> data = std::vector<float>(channels[i].reshape(1, resize_img.cols * resize_img.rows));
//		inputTensorValues.insert(inputTensorValues.end(), data.begin(), data.end());
//	}
//	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//	vector<Ort::Value> inputTensors;
//	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), input_dims.data(), input_dims.size()));
//	startTime = clock();
//	auto outputTensor = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), inputTensors.data(), 1, output_names.data(), 1);   // 开始推理
//	//endTime = clock();
//	//打印模型信息 /
//		//printf("Using Onnxruntime C++ API\n");
//		//printf("Number of inputs = %zu\n", num_input_nodes);
//		//printf("Number of output = %zu\n", num_output_nodes);
//		//std::cout << "input_name:" << input_name << std::endl;
//		//std::cout << "output_name: " << output_name << std::endl;
//		//std::cout << "input_dims:" << input_dims[0] << input_dims[1] << input_dims[2] << input_dims[3] << std::endl;
//		//std::cout << "output_dims:" << output_dims[0] << output_dims[1] << output_dims[2] << output_dims[3] << std::endl;
//		//std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//
//	//输出处理//
//	const int num_classes = 4;
//	float* prob = outputTensor[0].GetTensorMutableData<float>(); //outtensor首地址
//	int predict_label = std::max_element(prob, prob + num_classes) - prob;  // 确定预测类别的下标
//
//	float confidence = prob[predict_label];    // 获得预测值的置信度
//	printf("confidence = %f, label = %d\n", confidence, predict_label);
//	endTime = clock();//计时结束
//	std::cout << "total推理时间: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//	//vector< unsigned char >results(512 * 512);
//	//for (int i = 0; i < 512 * 512; i++)
//	//{
//	//	if (mask_ptr[i] >= 0.5)
//	//	{
//	//		results[i] = 0;
//	//	}
//	//	else
//	//	{
//	//		results[i] = 255;
//	//	}
//	//}
//	//unsigned char* ptr = &results[0];
//	//Mat mask = Mat(output_dims[2], output_dims[3], CV_8U, ptr);
//	//resize(mask, imgSrc, Size(imgBGR.cols, imgBGR.rows));
//	//原图展示分割结果//
//	//cvtColor(imgSrc, imgSrc, COLOR_GRAY2BGR);
//	//Mat imgAdd;
//	//addWeighted(imgBGR, 1, imgSrc, 0.3, 0, imgAdd);
//}

//智能方法分成.h之后测试
//#include "AIAlgorithmToDetectSolderJointDefects.h"
//int main()
//{
//	double he = 0;
//	cv::String pattern = "C:\\Users\\LENOVO\\Pictures\\image\\*.png";//测试图片路径
//	std::vector<std::string>img_path;
//	std::vector<cv::String>fn;
//	cv::glob(pattern, fn, false);
//	size_t count = fn.size();
//	for (size_t j = 0; j < count; j++)
//	{
//		img_path.push_back(fn[j]);
//	}
//
//	for (int i = 0; i < img_path.size(); i++)
//	{
//		std::string a = std::to_string(i + 1);// 1开始命名存储
//		cv::Mat input_img = cv::imread(img_path[i]);
//		//start = clock();//开始计时
//		clock_t startTime, endTime;
//		startTime = clock();
//		SegmentAIONNX(input_img, 300, 300);
//		endTime = clock();//计时结束
//		double tem = (static_cast<double>(endTime) - startTime) / CLOCKS_PER_SEC;
//		he += tem;
//
//		//center_resize(input_img);
//		//imwrite(a + ".png", input_img);
//	}
//	std::cout << "平均每张推理时间: " << he / fn.size() << "s" << std::endl;
//	return 0;
//
//}



//一键标定测试
//#include "CameraCalibration.h"
//#include <cmath>
//int main() 
//{
//    //一键标定
//    std::vector<cv::Point2f> phy;
//    std::vector<cv::Point2f> pix;
//    cv::Point2f a(-50,250);
//    cv::Point2f b(-50,270);
//    cv::Point2f c(-70,250);
//    
//    phy.push_back(a);
//    phy.push_back(c);
//    phy.push_back(b);
//
//    cv::Point2f d(1692, 1574);
//    cv::Point2f e(1671, 62);
//    cv::Point2f f(181, 1576);
//    pix.push_back(d);
//    pix.push_back(f);
//    pix.push_back(e);
//
//    caneraCalibration cal;
//    double cameraAngle = -1;
//    cv::Matx22d modelMartrix(0,0,0,0);
//    double xxx = cal.calibration(phy,pix,cameraAngle,modelMartrix);
//    std::cout << "pixel:" << xxx << std::endl;
//    std::cout << "cameraAngle:" << cameraAngle << std::endl;
//    std::cout << "juzhen:" << modelMartrix << std::endl;
//
//
//    return 0;
//}

//贴片中的小黑块检测
//#include <iostream>
//#include "blackBlock.h"
//
//int main() {
//    std::vector<std::string> path;
//    cv::glob("C:\\Users\\LENOVO\\Pictures\\smt\\*jpg",path);
//    for (int i = 0; i < path.size(); i++)
//    {
//        cv::Mat temp = cv::imread(path[i]);
//        getBlackBlock(temp);
//    }
//    return 0;
//}

//
////拼版框的模板匹配
//#include <iostream>
//#include "splicingBoard.h"
//
//void main() {
//    splicBoard sp;
//    cv::Mat image = cv::imread("C:\\Users\\LENOVO\\Pictures\\pinban.bmp");
//    std::vector<cv::Rect2f> rect;
//    //sp.matchingSplic(image,rect);
//    sp.linesMatchSplic(image, rect);
//}


//edline测试
//#include<iostream>
//#include"edline.h"
//#include <algorithm>
//#define WAIT_TIME 4000
//
//using namespace cv;
//
//
//int main(int argc, const char* argv[])
//{
//    Mat imageRGB = imread("C:/Users/LENOVO/Pictures/pinban.jpg");
//    Mat image = imread("C:/Users/LENOVO/Pictures/pinban.jpg", 0);
//    int x = 0, y = 0;
//    std::cout << " 输入规格(x * y) :" << std::endl;
//    std::cin >> x >> y;
//    EDLines ed;
//    std::vector<cv::Rect> roiRects = ed.getROIs(imageRGB,x,y);
//    for (cv::Rect roi : roiRects) {
//    cv::rectangle(imageRGB, roi, cv::Scalar(2, 2, 255), 2);
//    }
//    cv::namedWindow("test");
//    cv::imshow("test", imageRGB);//res
//    imwrite("./10_30.jpg", imageRGB);
//    cv::waitKey(0);

    //EDLines lineHandler = EDLines(image);
    //Mat outputImage;

    //namedWindow("INPUT IMAGE",0);
    //imshow("INPUT IMAGE", imageRGB);
    //waitKey(WAIT_TIME);
    //outputImage = lineHandler.getSmoothImage();
    //namedWindow("SMOOTHING", 0);
    //imshow("SMOOTHING", outputImage);
    //waitKey(WAIT_TIME);
    //outputImage = lineHandler.getGradImage();
    //namedWindow("GRADIENT AND THRESHOLDING", 0);
    //imshow("GRADIENT AND THRESHOLDING", outputImage);
    //waitKey(WAIT_TIME);
    //outputImage = lineHandler.getAnchorImage();
    //namedWindow("ANCHORING AND CONNECTING THEM", 0);
    //imshow("ANCHORING AND CONNECTING THEM", outputImage);
    //imwrite("./ANCHORING_AND_CONNECTING_THEM.jpg", outputImage);
    //waitKey(WAIT_TIME);
    //outputImage = lineHandler.getEdgeImage();
    //namedWindow("EDGES", 0);
    //imshow("EDGES", outputImage);
    //imwrite("./EDGES.jpg", outputImage);
    //waitKey(WAIT_TIME);
    //outputImage = lineHandler.getLineImage();
    //namedWindow("ED LINES", 0);
    //imshow("ED LINES", outputImage);
    //imwrite("./ED_LINES.jpg", outputImage);
    //waitKey(WAIT_TIME);


    /*std::vector<LS> lines, verticalLines;
    outputImage = lineHandler.drawOnImage(lines);*/

    //namedWindow("ED LINES OVER SOURCE IMAGE", 0);
    //imshow("ED LINES OVER SOURCE IMAGE", outputImage);
    //imwrite("./ED_LINES_OVER_SOURCE_IMAGE.jpg", outputImage);
    //waitKey(0);
    //std::cout << "size:" << lines.size()<<std::endl;
    //for (int i = 0; i < lines.size(); i++)
    //{
    //    if (std::abs(lines[i].start.x - lines[i].end.x) < 10 && std::abs(lines[i].start.y - lines[i].end.y) > 50) //说明是水平的
    //    {
    //        verticalLines.push_back(lines[i]);
    //    }
    //}
    //std::cout << "---------------------" << verticalLines.size() << std::endl;
    // 
    //cv::Mat res = lineHandler.drawOwn(verticalLines);
    //namedWindow("test", 0);
    //imshow("test", imageRGB);//res
    ////imwrite("./test.jpg", res);
    //waitKey(0);

    //std::vector<int> xCoordinates;

    //// 遍历线段集合，获取所有X坐标
    //for (const auto& ls : lines) {
    //    xCoordinates.push_back(ls.start.x);
    //    xCoordinates.push_back(ls.end.x);
    //}

    //// 对X坐标进行排序
    //std::sort(xCoordinates.begin(), xCoordinates.end());//, compare

    //int maxGap = 0;
    //int densestX = 0;

    //// 寻找最大间隔并计算最密集的X坐标
    //for (int i = 0; i < xCoordinates.size() - 1; i++) {
    //    int gap = xCoordinates[i + 1] - xCoordinates[i];
    //    if (gap > maxGap) {
    //        maxGap = gap;
    //        densestX = (xCoordinates[i] + xCoordinates[i + 1]) / 2;
    //    }
    //}

    //std::cout << "最密集的X坐标是: " << densestX << std::endl;

//    return 0;
//}

//bool compare(int a, int b) {
//    return a < b;
//}


//halcontest
//#include"halconMatch.h"
//#include<time.h>
//#include <halconcpp/HalconCpp.h>
//#include "Halcon.h"
//#include <iostream>
//#include <opencv2/opencv.hpp>

//int main() {
    //// Local iconic variables
    //HObject  ho_DSC_0097;

    //// Local control variables
    //HTuple  hv_WindowHandle;

    //ReadImage(&ho_DSC_0097, "C://Users//LENOVO//Pictures//test_rpc.jpg");
    //dev_open_window_fit_image(ho_DSC_0097, 0, 0, -1, -1, &hv_WindowHandle);
    //if (HDevWindowStack::IsOpen())
    //    ClearWindow(HDevWindowStack::GetActive());
    //if (HDevWindowStack::IsOpen())
    //    DispObj(ho_DSC_0097, HDevWindowStack::GetActive());
    //        system("pause");
    //std::cout << "---" << std::endl;
    //return 0;


  // Local iconic variables
//    HalconCpp::HObject  ho_Image, ho_ModelRegion, ho_TemplateImage;
//    HalconCpp::HObject  ho_TransContours, ho_RegionAffineTrans;
//
//    // Local control variables
//    HalconCpp::HTuple  hv_ModelID, hv_ModelRegionArea, hv_RefRow;
//    HalconCpp::HTuple  hv_RefColumn, hv_TestImages, hv_T, hv_Row, hv_Column;
//    HalconCpp::HTuple  hv_Angle, hv_Score, hv_I, hv_AlignmentHomMat2D;
//    HalconCpp::HTuple  models;
//    //
//    //Matching 01: ************************************************
//    //Matching 01: BEGIN of generated code for model initialization
//    //Matching 01: ************************************************
//    //
//    //Matching 01: Obtain the model image
//    ReadImage(&ho_Image, "E:/pycharmWP/halcon/pinbXh.jpg");
//    //
//    //Matching 01: Build the ROI from basic regions
//    GenRectangle1(&ho_ModelRegion, 40.0014, 172.364, 245.532, 325.054);
//    //
//    //Matching 01: Reduce the model template
//    ReduceDomain(ho_Image, ho_ModelRegion, &ho_TemplateImage);
//    //
//    //Matching 01: Create the correlation model
//    CreateNccModel(ho_TemplateImage, "auto", HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
//        "auto", "use_polarity", &hv_ModelID);
//    //
//    //Matching 01: Get the reference position
//    AreaCenter(ho_ModelRegion, &hv_ModelRegionArea, &hv_RefRow, &hv_RefColumn);
//    //
//    //Matching 01: Display the model region
//    /*if (HalconCpp::HDevWindowStack::IsOpen())
//        DispObj(ho_Image, HalconCpp::HDevWindowStack::GetActive());
//    if (HalconCpp::HDevWindowStack::IsOpen())
//        HalconCpp::SetColor(HalconCpp::HDevWindowStack::GetActive(), "green");
//    if (HalconCpp::HDevWindowStack::IsOpen())
//        HalconCpp::SetDraw(HalconCpp::HDevWindowStack::GetActive(), "margin");
//    if (HalconCpp::HDevWindowStack::IsOpen())
//        DispObj(ho_ModelRegion, HalconCpp::HDevWindowStack::GetActive());
//    GenCrossContourXld(&ho_TransContours, hv_RefRow, hv_RefColumn, 20, 0.0);
//    if (HalconCpp::HDevWindowStack::IsOpen())
//        DispObj(ho_TransContours, HalconCpp::HDevWindowStack::GetActive());*/
//    // stop(...); only in hdevelop
//    //
//    //Matching 01: END of generated code for model initialization
//    //Matching 01:  * * * * * * * * * * * * * * * * * * * * * * *
//    //Matching 01: BEGIN of generated code for model application
//    //
//    //Matching 01: Loop over all specified test images
//    hv_TestImages = "E:/pycharmWP/halcon/pinbXh.jpg";
//    for (hv_T = 0; hv_T <= 0; hv_T += 1)
//    {
//        //
//        //Matching 01: Obtain the test image
//        ReadImage(&ho_Image, HalconCpp::HTuple(hv_TestImages[hv_T]));
//        //
//        //Matching 01: Find the modelImage（in）：单通道图像，它的区域可被创建为模板
////ModelID（in）：模板句柄         AngleStart（in）：模板的最小旋转        AngleExtent（in）：旋转角度范围          MinScore（in）：被找到的模板最小分数 
////NumMatches（in）：被找到的模板个数          MaxOverlap（in）：被找到的模板实例最大重叠部分           SubPixel（in）：亚像素级别标志,true, false     
////NumLevels（in）：金字塔层级数          Row（out）：被找到的模板实例行坐标
////Column（out）：被找到的模板实例列坐标       Angle（out）：被找到的模板实例的旋转角度        Score（out）：被找到的模板实例的分数
//        FindNccModel(ho_Image, hv_ModelID, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
//            0.5, 3, 0.5, "true", 0, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);
//        for (int i = 0; i < hv_Row.Length(); i++)
//        {
//            std::cout << i << ":" << hv_Row[i].D() << " " << hv_Column[i].D() << " " << hv_Score[i].D() << std::endl;
//        }
//        
//
//        //Matching 01: Display the centers of the matches in the detected positions
//        //if (HDevWindowStack::IsOpen())
//        //    DispObj(ho_Image, HDevWindowStack::GetActive());
//        //{
//        //    HTuple end_val45 = (hv_Score.TupleLength()) - 1;
//        //    HTuple step_val45 = 1;
//        //    for (hv_I = 0; hv_I.Continue(end_val45, step_val45); hv_I += step_val45)
//        //    {
//        //        //Matching 01: Display the center of the match
//        //        GenCrossContourXld(&ho_TransContours, HTuple(hv_Row[hv_I]), HTuple(hv_Column[hv_I]),
//        //            20, hv_Angle);
//        //        if (HDevWindowStack::IsOpen())
//        //            SetColor(HDevWindowStack::GetActive(), "green");
//        //        if (HDevWindowStack::IsOpen())
//        //            DispObj(ho_TransContours, HDevWindowStack::GetActive());
//        //        HomMat2dIdentity(&hv_AlignmentHomMat2D);
//        //        HomMat2dTranslate(hv_AlignmentHomMat2D, -hv_RefRow, -hv_RefColumn, &hv_AlignmentHomMat2D);
//        //        HomMat2dRotate(hv_AlignmentHomMat2D, HTuple(hv_Angle[hv_I]), 0, 0, &hv_AlignmentHomMat2D);
//        //        HomMat2dTranslate(hv_AlignmentHomMat2D, HTuple(hv_Row[hv_I]), HTuple(hv_Column[hv_I]),
//        //            &hv_AlignmentHomMat2D);
//        //        //Matching 01: Display the aligned model region
//        //        AffineTransRegion(ho_ModelRegion, &ho_RegionAffineTrans, hv_AlignmentHomMat2D,
//        //            "nearest_neighbor");
//        //        if (HDevWindowStack::IsOpen())
//        //            DispObj(ho_RegionAffineTrans, HDevWindowStack::GetActive());
//        //        //Matching 01: Insert your code using the alignment here, e.g. code generated by
//        //        //Matching 01: the measure assistant with the code generation option
//        //        //Matching 01: 'Alignment Method' set to 'Affine Transformation'.
//        //        // stop(...); only in hdevelop
//        //    }
//        //}
//    }
//    //
//    //Matching 01: *******************************************
//    //Matching 01: END of generated code for model application
//    //Matching 01: *******************************************
//    //
//    ////cv::Mat res = HImageToMat(ho_TransContours);
//    ////cv::Mat res = HObject2Mat(ho_TransContours);
//    //cv::Mat res;
//    //HObject2MatImg(ho_TransContours,res);
//    //cv::namedWindow("test11", 0);
//    //cv::imshow("test11", res);//res
//    ////imwrite("./test.jpg", res);
//    //cv::waitKey(0);
//    ////
//    ////Matching 01: *******************************************
//    ////Matching 01: END of generated code for model application
//    ////Matching 01: *******************************************
//    ////


//    HMatch hm;
//    clock_t sttime, endtime;
//    sttime = clock();
//    //cv::Mat Image = cv::imread("E:\\pycharmWP\\halcon\\hui.jpg");
//    //cv::Rect2f rect(6820,1518,5553,8230);
//    //std::vector<cv::Rect2f> res;
//    //hm.getPanelFrames(3,Image,rect,res);
//    cv::Mat Image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask1_big.jpg");
//    cv::Rect2f rect(658, 594, 341, 337);//999 931
//    std::vector<cv::Rect2f> res;
//    hm.getPanelFrames(1, Image, rect, res);
//    endtime = clock();
//    std::cout << "model init use:" << (static_cast<double>(endtime) - sttime) / CLOCKS_PER_SEC << "s" << std::endl;
//    for (int i = 0; i < res.size(); i++)
//    {
//        cv::rectangle(Image, res[i], cv::Scalar(2, 2, 255), 6);
//    }
//    cv::rectangle(Image,rect, cv::Scalar(255, 255, 255), 12);
//    //cv::imwrite("./pin.jpg", Image);
//    return 0;
//}


//patchDetection
//#include "patchDetection.h"
//#include "solderDefectAI.h"
//#include"onnx.h"
//#include "fov_puzzle.h"
//#include "fovRoutePlan.h"
//#include "fovTest.h"
//#include"edline.h"
//int main() {
    // 载入原图和模板图
    //cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\black\\1699863082241.png", cv::IMREAD_COLOR);
    //cv::Mat templ = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\black\\1699863164939.png", cv::IMREAD_COLOR);

    //是否错件
    //pathDetection pa;
    //std::cout<<pa.ifDetectionWrong(img,templ)<<std::endl;

    //这一段是检测小黑块是否存在
    //// 检查图片是否正确载入
    //if (img.empty() || templ.empty()) {
    //    std::cout << "Could not read the image." << std::endl;
    //    return 1;
    //}

    //// 进行模板匹配
    //patchAlgorithm pA;
    ////bool isMatchFound = pA.ifBlackBlockExist(img, templ) || pA.ifBlackBlockExist(img, pA.rotateImg(templ, 90));

    //bool isMatchFound = pA.variousAngleExist(img, templ, 90);
    //// 输出结果
    //if (isMatchFound) {
    //    std::cout << "Match found." << std::endl;
    //}
    //else {
    //    std::cout << "No match found." << std::endl;
    //}

    //偏移角度
    //pathDetection pa;
    //std::cout << pa.detectOffsetAngle(img) << std::endl;

    //edline方法
    //EDLines lineHandler = EDLines(img);
    //cv::Mat outputImage;
    //std::vector<LS> lines, verticalLines;
    //outputImage = lineHandler.drawOnImage(lines);

    //cv::namedWindow("1114", 0);
    ////cv::imshow("1114", outputImage);
    //cv::imwrite("./1114.jpg", outputImage);
    ////cv::waitKey(0);



    //找黑块
    //std::vector<std::string> path;
    //clock_t start, end;
    //cv::glob("C:\\Users\\LENOVO\\Pictures\\smt\\black\\*png",path);
    //for (int i = 0; i < path.size(); i++)
    //{
    //    cv::Mat temp = cv::imread(path[i]);
    //    start = clock();
    //    pa.getBlackBlock(temp);
    //    end = clock();
    //    std::cout << "model init use:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << "s" << std::endl;
    //}

    //修改后的焊点智能方法测试
 //   double he = 0;
 //   std::string model_path = "acc-76.onnx";
 //   clock_t sttime, endtime;
 //   sttime = clock();
 //   OnnxModelAI model(model_path);
 //   endtime = clock();
 //   std::cout << "model init use:" << (static_cast<double>(endtime) - sttime) / CLOCKS_PER_SEC << "s" << std::endl;
	//cv::String pattern = "C:\\Users\\LENOVO\\Pictures\\image\\*.png";//测试图片路径
	//std::vector<std::string>img_path;
	//std::vector<cv::String>fn;
	//cv::glob(pattern, fn, false);
	//size_t count = fn.size();
	//for (size_t j = 0; j < count; j++)
	//{
	//	img_path.push_back(fn[j]);
	//}

	//for (int i = 0; i < img_path.size(); i++)
	//{
	//	std::string a = std::to_string(i + 1);// 1开始命名存储
	//	cv::Mat input_img = cv::imread(img_path[i]);
 //       clock_t startTime, endTime;
 //       startTime = clock();
 //       auto result = model.predict(input_img);
 //       endTime = clock();//计时结束
 //       double tem = (static_cast<double>(endTime) - startTime) / CLOCKS_PER_SEC;
	//	he += tem;
 //       std::cout << "Label: " << result.first << ", Confidence: " << result.second << ", time:"<<tem << std::endl;
	//	//center_resize(input_img);
	//}
 //   std::cout << "平均每张推理时间: " << he / fn.size() << "s" << std::endl;


    //std::string model_path = "SmtAi.onnx";
    //OnnxModelAI model(model_path);

    //cv::Mat imgSrc = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\3.jpg"); // 输入图像
    //auto result = model.predict(imgSrc);

    //std::cout << "Label: " << result.first << ", Confidence: " << result.second << std::endl;

    //fovrouteplan测试
    ////获取一下坐标
    //Net_config yolo_nets = { 0.4, 0.4, 0.4,"best6.onnx" };//best_rpc.onnx   10.12_rpc
    //YOLO yolo_model(yolo_nets);

  
    //vector<cv::Rect2f> smt_frame;
   
    //cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask3.jpg");
    //smt_frame = yolo_model.detect(img1);
    //
    //for (auto dian : smt_frame)
    //{
    //    cv::rectangle(img1, dian, cv::Scalar(2, 255, 255), 10);
    //}

    //fovRoute fr;
    //// 调用函数
    //
    //std::vector<cv::Point> temp;
    //for (int i = 0; i < smt_frame.size(); i++)
    //{
    //    cv::Point2f lalala(smt_frame[i].x,smt_frame[i].y);
    //    temp.push_back(lalala);
    //}

    //std::vector<cv::Rect> result = fr.coverPointsWithRectangles(temp);
    //std::cout << "result:" << result.size() << std::endl;
    //for (auto& rect : result)
    //{
    //    cv::rectangle(img1, rect, cv::Scalar(2, 2, 255), 20);
    //}
    //cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1122.jpg", img1);

    //fovTest 测试
//    Net_config yolo_nets = { 0.4, 0.4, 0.4,"best6.onnx" };//best_rpc.onnx   10.12_rpc
//    YOLO yolo_model(yolo_nets);
//
//
//    vector<cv::Rect2f> smt_frame;
//
//    cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\1\\1\\1.jpg");
//    smt_frame = yolo_model.detect(img1);
//
//    //std::vector<cv::Point> points;
//
//    for (auto dian : smt_frame)
//    {
//        //points.push_back(cv::Point(dian.x, dian.y));
//        cv::rectangle(img1, dian, cv::Scalar(2, 255, 255), 10);
//    }
//
//    FovRoute fov_route(1, 1, 1, 1);
//    //std::vector<cv::Point> points = { {100, 150}, {200, 250}, {300, 350} };
//    int width = 2448;
//    int height = 2048;
//    cv::Size imageSize = img1.size();
//    std::vector<cv::Rect> rectangles = fov_route.coverPointsWithRectangles(smt_frame, width, height,imageSize);
//
//    for (const auto& rect : rectangles) {
//        if (rect.x < 0 || rect.y < 0)
//        {
//            rectangles.pop_back();
//            continue;
//        }
//        cv::rectangle(img1, rect, cv::Scalar(2, 2, 255), 20);
//    }
//    std::cout << rectangles.size() << std::endl;
//   
//    cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1122TestRect.jpg", img1);

//小黑块坐标转化
    //Net_config yolo_nets = { 100, 0.4, 0.4,"12_21onlyrc.onnx" };//best_rpc.onnx   10.12_rpc
    //YOLO yolo_model(yolo_nets);

    ////std::vector<std::vector<cv::Rect2f>> res(2);
    //

    //clock_t startTime1, endTime1,startTime2,endTime2;
    //
    //cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\1\\1\\1.jpg");
    //startTime1 = clock();
    ////分割
    //std::vector<cv::Rect2f> temp = pa.splitImage(img1.size());//得到分割的小图的rect
    //endTime1 = clock();
    //double tem = (static_cast<double>(endTime1) - startTime1) / CLOCKS_PER_SEC;
    //std::cout <<"分割图片:" << tem << std::endl;
    //startTime2 = clock();
    //for (auto rect : temp)
    //{
    //    std::vector<std::vector<cv::Rect2f>> smt_frame;
    //    cv::Mat smallImg = img1(rect);
    //    clock_t sta, endd;
    //    sta = clock();
    //    smt_frame = yolo_model.detect(smallImg);
    //    
    //    endd = clock();
    //    std::cout << "小图:" << (static_cast<double>(endd) - sta) / CLOCKS_PER_SEC << std::endl;;
    //    cv::rectangle(img1, rect, cv::Scalar(2, 255, 255), 10);
    //    if (smt_frame.size() != 0) 
    //    {
    //        std::cout << smt_frame[0].size() << std::endl;
    //        std::cout << smt_frame[1].size() << std::endl;
    //        for (int i = 0; i < smt_frame.size(); i++)
    //        {
    //            for (int j = 0; j < smt_frame[i].size(); j++)
    //            {
    //                smt_frame[i][j].x += rect.x;
    //                smt_frame[i][j].y += rect.y;
    //                //res.push_back(smt_frame[i]);
    //            }

    //        }
    //        /*if (smt_frame[0].size() != 0)
    //        {*/
    //        for (int x = 0; x < smt_frame[0].size(); x++)
    //        {
    //            res[0].push_back(smt_frame[0][x]);
    //        }
    //        for (int y = 0; y < smt_frame[1].size(); y++)
    //        {
    //            res[1].push_back(smt_frame[1][y]);
    //        }
    //        //}
    //    }
    //    
    //   
    //}
    //endTime2 = clock();
    //double tem1 = (static_cast<double>(endTime2) - startTime2) / CLOCKS_PER_SEC;
    //std::cout << "得到坐标:" << tem1 << std::endl;
    
    //fov_scan fs;
    //// 读取图片
    //cv::Mat img11 = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask1.jpg", cv::IMREAD_GRAYSCALE);
    //cv::Mat img22 = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask2.jpg", cv::IMREAD_GRAYSCALE);
    //fs.getSHift(img11,img22);
    //fs.getCircleCenter(img1);
//    fs.getShiftImage(img1,50,50);
//


    /*std::vector<std::vector<cv::Rect2f>> res;
    res = yolo_model.getCPCoordinate(img1);

        for (auto j : res[0])
        {
            cv::rectangle(img1, j, cv::Scalar(2, 255, 255), 10);
        }  
        for (auto j : res[1])
        {
            cv::rectangle(img1, j, cv::Scalar(255, 255, 0), 10);
        }  
*/
//    return 0;
//}


//#include "colorMethod.h"
//#include <time.h>
//int main() 
//{
//   /* clock_t start, end, start1, end1;
//    colorMethod clo;
//    cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\6.jpg");
//    start = clock();
//    clo.test(img);
//    end = clock();
//
//    start1 = clock();
//    clo.test1(img);
//    end1 = clock();
//
//    std::cout<< (static_cast<double>(end) - start) / CLOCKS_PER_SEC <<std::endl;
//    std::cout<< (static_cast<double>(end1) - start1) / CLOCKS_PER_SEC  <<std::endl;*/
//
//
//    colorMethod clo;
//    clo.test2();
//    return 0;
//}

//unet测试
//#include "uNetOnnx.h"
//#include <time.h>
//int main() 
//{
    //// 模型路径和输入图像数据
    //std::string model_path = "unetmodel.onnx";
    //cv::Mat image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\2499.jpg", cv::IMREAD_COLOR);
    //std::vector<int64_t> input_shape = { 1, 3, 512, 512 }; // 根据需要调整尺寸

    //// 创建一个模型推断器
    //uNetOnnx inferer(model_path);

    //std::vector<float> input_data = inferer.getInputData(image);; // 用图像数据填充这个
    //// 运行推断
    //std::vector<float> output_data = inferer.RunInference(input_data, input_shape);

    //// 输出结果，或者进一步处理
    //cv::Mat res = inferer.getImageFromFloat(output_data);
    //cv::imwrite("C:\\Users\\LENOVO\\Pictures\\smt\\niubi.jpg",res);

//    cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\3.png");//2499
//    std::string modelFile = "model_v2_sim.onnx";//unetmodel model_v2_sim
//
//    clock_t start, endd;
//    start = clock();
//    /*cv::Mat imgg = inferer.dnnGetClasses(img,modelFile);
//    inferer.getMultipodPackRect(imgg);*/
//    uNetOnnx aaa;
//    cv::dnn::Net bbbb= aaa.loadUnetModel(modelFile);
//    std::vector<cv::Rect> ccccc = aaa.getDnnClassesRect(img, bbbb);
//    endd = clock();
//    std::cout << (static_cast<double>(endd) - start) / CLOCKS_PER_SEC << std::endl;
//    return 0;
//}


//mark点根据halcon算偏移
//#include"halconMatch.h"
//#include<time.h>
//void main() 
//{
//    HMatch hm;
//    clock_t sttime, endtime;
//    sttime = clock();
//    cv::Mat Image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\fov1.jpg");
//    cv::Mat mask = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask1.jpg");
//    //cv::Rect2f rect(658, 594, 341, 337);//999 931
//    cv::Rect2f res;//找到的位置rect
//    hm.getMaskShift(1, Image, mask, res);
//    endtime = clock();
//    std::cout << "model init use:" << (static_cast<double>(endtime) - sttime) / CLOCKS_PER_SEC << "s" << std::endl;
//    
//    cv::rectangle(Image, res, cv::Scalar(2, 2, 255), 6);
//   
//}

//yolo预测的时候用颜色阈值同时得到小黑块的坐标  可以用颜色方法得到子框，也可用unet
//#include "onnx.h"
//
//int main() 
//{
//    //初始化一下unet模型
//    uNetOnnx aaa;
//    std::string modelFile = "unetmodel.onnx";
//    cv::dnn::Net bbbb = aaa.loadUnetModel(modelFile);
//
//    bool useUnet = true;
//
//    Net_config yolo_nets = { 0.4, 0.4, 0.55,"best_smd.onnx" };//bestrpc.onnx   10.12_rpc
//    YOLO yolo_model(yolo_nets);
//    cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\pinban.jpg");//image\\1.jpg
//
//    clock_t start, endd;
//    start = clock();
//
//    auto smt_frame = yolo_model.detect(img1,bbbb,useUnet);
//    
//    endd = clock();
//    std::cout << (static_cast<double>(endd) - start) / CLOCKS_PER_SEC << std::endl;
//
//    for (auto dian : smt_frame)
//    {
//        cv::rectangle(img1, dian.first, cv::Scalar(2, 255, 255), 10);
//        cv::rectangle(img1, dian.second, cv::Scalar(255, 255, 0), 10);
//    }
//    cv::imwrite("C:\\Users\\LENOVO\\Pictures\\smt\\01_23.jpg", img1);
//    return 0;
//}

//检测时候图片的旋转
//#include "fov_puzzle.h"
//int main() 
//{
//    fov_scan fs;
//    //这两行是旋转图片
// /*   cv::Mat Image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\fov1.jpg");
//    auto roateImage = fs.RotateImg(Image,-0.8);*/
//
//    //根据两个点的偏移数据算平移量和旋转角度
//    std::pair<int, int> offset1 = { -18, 577 };
//    std::pair<int, int> offset2 = { -28, 588 };
//
//    Transform result = fs.calculateTransform(offset1, offset2);
//    std::cout << "Rotation Angle: " << result.rotationAngle << " degrees" << std::endl;
//    std::cout << "Translation: (" << result.translation.first << ", " << result.translation.second << ")" << std::endl;
//    return 0;
//}


//bisenet测试-onnx
//#include "biSeNetOpenvion.h"
////#include <openvino/openvino.hpp>
////#include <iostream>
//int main() 
//{
//    biSeNetOpenvino bisenet;
//    bisenet.test();
//    //ov::Core ie;
//    ////获取当前支持的所有的AI硬件推理设备
//    //std::vector<std::string> devices = ie.get_available_devices();
//    //for (int i = 0; i < devices.size(); i++) {
//    //    std::cout << devices[i] << std::endl;
//    //}
//    return 0;
//}


//优化yolo 思路 消除背景影响
#include "eliminateYoloBackground.h"
#include "patchDetection.h"
#include "halconMatch.h"
#include <time.h>
#include <filesystem>
#include "onnx2.h"

namespace fs = std::filesystem;

int main() 
{
    eliminateYoloBackground e;
    // Load your image C://Users//LENOVO//Pictures//smt//compimg//compimg//0.jpg
    cv::Mat image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\289.jpg");//C:\\Users\\LENOVO\\Pictures\\smt\\4.jpg

 //   // Define the color range for BFS
 //   int colorRange = 10; 

    clock_t start,end;
 //   start = clock();
 //   // Get the bounding rectangle
 //   auto [topLeft, bottomRight] = e.findBoundingRectangle(image, colorRange);
 //   end = clock();
 //   std::cout <<"bfs:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;

 //   // Draw the bounding rectangle on the image
 //   cv::rectangle(image, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2); // Green rectangle with thickness 2

 ///*   auto rect_of_yinjiao = e.findPinRectangles(image);
 //   for (auto &rect : rect_of_yinjiao)
 //   {
 //       cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
 //   }*/
 //   cv::Rect recttt = cv::Rect(topLeft.x,topLeft.y,bottomRight.x - topLeft.x,bottomRight.y - topLeft.y);

 //   start = clock();
 //   auto aaa = e.findPinsAroundBlackBox(image, recttt);
 //   end = clock();
 //   std::cout << "bfs of pin:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
 //   for (auto& rect : aaa)
 //   {
 //       cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
 //   }

 //   //全部筛选的方法
 //   pathDetection p;
 //   start = clock();
 //   auto res = p.getBlackBlock(image);
 //   end = clock();
 //   std::cout << "all:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
 //   cv::rectangle(image, res, cv::Scalar(255, 255, 255), 2);


    
    //轮廓信息测试---单张图片测试---中心点向外扩散，碰到边界和某条边白色像素占比超过阈值停止
    //start = clock();
    //cv::Mat heibai = e.test(image);
    //end = clock();
    //std::cout << "ini:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
    //start = clock();
    //auto [topLeft, bottomRight] = e.findBoundingRectangle_heibai(heibai,0.2);//0.2
    //end = clock();
    //std::cout << "bfs:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;

    //cv::Mat hsvImage = e.useHsvTest(image);//用hsv处理的图做一下交集
    //cv::rectangle(image, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);

    //cv::Rect2f recttt = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    //start = clock();
    //auto aaa = e.findPinsAroundBlackBox(heibai, recttt, hsvImage);
    //end = clock();
    //std::cout << "bfs of pin:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
    //for (auto& rect : aaa)
    //{
    //    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
    //}


    //轮廓信息测试--多张图片测试---中心点向外扩散，碰到边界和某条边白色像素占比超过阈值停止

    //std::string inputDirectory = "F:\\AOI-namespace\\多引脚\\多引脚\\";
    //std::string outputDirectory = "C:\\Users\\LENOVO\\Pictures\\smt\\test\\";

    //// 创建输出目录
    //fs::create_directory(outputDirectory);

    //std::vector<std::string> fileNames;
    //cv::glob(inputDirectory + "*.jpg", fileNames); // 获取所有.png文件

    //double totalIniTime = 0;
    //double totalBfsTime = 0;
    //double totalProcessTime = 0;
    //int imageCount = 0;

    //for (const auto& fileName : fileNames) {
    //    cv::Mat image = cv::imread(fileName);

    //    // 开始处理图片并计时
    //    clock_t start, end;
    //    start = clock();
    //    cv::Mat heibai = e.test(image); 
    //    end = clock();
    //    double iniTime = (static_cast<double>(end) - start) / CLOCKS_PER_SEC;
    //    totalIniTime += iniTime;

    //    std::cout << "ini:" << iniTime << std::endl;
    //    start = clock();
    //    auto [topLeft, bottomRight] = e.findBoundingRectangle_heibai(heibai, 0.2);
    //    end = clock();
    //    double bfsTime = (static_cast<double>(end) - start) / CLOCKS_PER_SEC;
    //    totalBfsTime += bfsTime;

    //    std::cout << "bfs:" << bfsTime << std::endl;

    //    // 绘制矩形并保存图像
    //    cv::rectangle(image, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);
    //    cv::imwrite(outputDirectory + fs::path(fileName).filename().string(), image);

    //    // 计算并打印每张图片的总处理时间
    //    double processTime = iniTime + bfsTime;
    //    totalProcessTime += processTime;
    //    std::cout << "Processing time for image " << fileName << ": " << processTime << " seconds." << std::endl;

    //    imageCount++;
    //}

    //// 计算平均处理时间
    //if (imageCount > 0) {
    //    std::cout << "Average ini time: " << totalIniTime / imageCount << std::endl;
    //    std::cout << "Average bfs time: " << totalBfsTime / imageCount << std::endl;
    //    std::cout << "Average total processing time per image: " << totalProcessTime / imageCount << std::endl;
    //}

    //单张调用函数测试
    
    //start = clock();
    //auto res = e.getBoundAndPin(image,"T");//T/R/D
    //end = clock();
    //std::cout << "all:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
    //cv::rectangle(image, res.first, cv::Scalar(0, 255, 255), 4);
    //for (int i = 0; i < res.second.size(); i++)
    //{
    //    cv::rectangle(image, res.second[i], cv::Scalar(0, 255, 0), 4);
    //}




    //hsv筛选原图引脚测试
    //e.useHsvTest(image);

    /*算拼版数量*/
    //std::vector<cv::Rect2f> res;
    //HMatch halcon;
    //cv::Mat imagegg = cv::imread("C:\\Users\\LENOVO\\Pictures\\pinban.jpg");
    //cv::Rect2f firstRect(843,1530,5941,8206);
    //int aaa = halcon.getNumberOfPanel(firstRect, imagegg.size());
    //halcon.getPanelFrames(aaa,imagegg,firstRect,res);
    ////cv::rectangle(imagegg, firstRect, cv::Scalar(0, 255, 0), 10);
    //for (auto temp : res)
    //{
    //    cv::rectangle(imagegg, temp, cv::Scalar(0, 255, 0), 10);
    //}

    /*黑色掩膜覆盖*/
    //cv::Mat heibai = e.test(image);
    //auto [topLeft, bottomRight] = e.findBoundingRectangle_heibai(heibai, 0.01);
    //cv::Rect2f recttt = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    //cv::rectangle(image, recttt, cv::Scalar(0, 0, 0), cv::FILLED);
    //cv::Mat gray;
    //cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    //
    //cv::Mat topImg = gray({0,0,static_cast<int>(image.cols),static_cast<int>(recttt.y)});
    //cv::Mat bottomImg = gray({0,static_cast<int>(recttt.y + recttt.height),static_cast<int>(image.cols),static_cast<int>(image.rows - (recttt.y + recttt.height))});
    //cv::threshold(topImg, topImg, 0, 255, cv::THRESH_TRIANGLE);
    //cv::threshold(bottomImg, bottomImg, 0, 255, cv::THRESH_TRIANGLE);
    //std::vector<std::vector<cv::Point>> contours;

    //cv::findContours(topImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //// 计算所有轮廓的最小外接矩形
    //cv::Rect boundingBox;
    //for (size_t i = 0; i < contours.size(); i++) {
    //    boundingBox |= cv::boundingRect(contours[i]);
    //}

    //cv::rectangle(topImg, boundingBox, cv::Scalar(255), 2);
    //cv::rectangle(image, boundingBox, cv::Scalar(0, 255, 0), 4);

    //contours.clear();

    //cv::findContours(bottomImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //cv::Rect boundingBox1;
    //for (size_t i = 0; i < contours.size(); i++) {
    //    boundingBox1 |= cv::boundingRect(contours[i]);
    //}

    //// 绘制外接矩形
    //cv::rectangle(bottomImg, boundingBox1, cv::Scalar(255), 2);
    //cv::rectangle(image, { boundingBox1.x, static_cast<int>(boundingBox1.y + (recttt.y + recttt.height)) ,boundingBox1 .width,boundingBox1.height}, cv::Scalar(0, 255, 0), 4);
    
    /*用模型测试分类以及检测效果*/
    Net_config yolo_nets = { 0.4, 0.4, 0.4,"best_rtp.onnx" };//bestrpc.onnx   10.12_rpc
    YOLO yolo_model(yolo_nets);
    cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\3_29\\2\\thumb.jpg");//image\\1.jpg
    start = clock();
    auto smt_frame = yolo_model.getCPCoordinate(img1);
    end = clock();
    //cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1\\1\\5.jpg", img1);
    std::cout << "all:" << (static_cast<double>(end) - start) / CLOCKS_PER_SEC << std::endl;
    int nums = 0;
    for (int i = 0; i < smt_frame.size(); i++)
    {
        cv::rectangle(img1, smt_frame[i].first, cv::Scalar(0, 255, 0), 4);
        for (int j = 0; j < smt_frame[i].second.size(); j++)
        {
            if (smt_frame[i].second[j].x > 0 && smt_frame[i].second[j].y > 0 && smt_frame[i].second[j].width > 0 && smt_frame[i].second[j].height > 0)
            {
                cv::rectangle(img1, smt_frame[i].second[j], cv::Scalar(255, 255, 0), 4);
            }
            else
            {
                nums++;
            }
        }
    }
    std::cout << "超出范围:" <<nums<< std::endl;
    //cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1\\1\\6.jpg", img1);
    return 0;
}