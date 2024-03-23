//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc_c.h>
//#include <opencv2/dnn.hpp>
//#include <opencv2/core/utils/logger.hpp>
//#include <iostream>
//#include <onnxruntime_cxx_api.h>
////#include <assert.h>
//#include "time.h"
//#include <vector>
//#include <fstream>
//#include <exception>
//#include <filesystem>
//namespace fs = std::filesystem;
//
//using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
//using namespace std;
//using namespace Ort;
//using namespace cv::dnn;
//
///*******************************************************************************/
////如果使用CUDA加速
////#define _CUDA_
////图像的宽和高
//#define IMG_H 128
//#define IMG_W 32
////种类
//#define TOTAL_CATEGORY 95
//String labels_txt_file = "F:\\OCR\\onnxruntime_inference_PARSeq\\onnxruntime_inference_PARSeq\\workspace\\char.txt";
//vector<String> readClassNames();
////testDatasets path
//const string Path{ "C:\\Users\\LENOVO\\Pictures\\smt\\center\\" };
//
//
//int _count_{};
//// 图像处理  标准化处理
//void PreProcess(const Mat& image, Mat& image_blob)
//{
//	Mat input;
//	image.copyTo(input);
//	//数据处理 标准化
//	std::vector<Mat> channels, channel_p;
//	split(input, channels);
//	Mat R, G, B;
//	B = channels.at(0);
//	G = channels.at(1);
//	R = channels.at(2);
//
//	B = (B / 255. - 0.5) / 0.5;
//	G = (G / 255. - 0.5) / 0.5;
//	R = (R / 255. - 0.5) / 0.5;
//
//	channel_p.push_back(R);
//	channel_p.push_back(G);
//	channel_p.push_back(B);
//
//	Mat outt;
//	merge(channel_p, outt);
//	image_blob = outt;
//}
//
//// 读取txt文件
//std::vector<String> readClassNames()
//{
//	std::vector<String> classNames;
//
//	std::ifstream fp(labels_txt_file);
//	if (!fp.is_open())
//	{
//		std::wcout << "could not open file..." << std::endl;
//		exit(-1);
//	}
//	std::string name;
//	while (!fp.eof())
//	{
//		std::getline(fp, name);
//		if (name.length())
//			classNames.push_back(name);
//	}
//	fp.close();
//	return classNames;
//}
//
//double Total_time{}, Ave_time{};
//string label{};
//
//int main()         // 返回值为整型带参的main函数. 函数体内使用或不使用argc和argv都可
//{
//
//	std::vector<String> src_test;
//	glob(Path, src_test, false);
//	if (src_test.size() == 0)
//	{//�ж��ļ��������Ƿ���ͼƬ
//		printf("error!!!\n");
//		exit(1);
//	}
//	_count_ = src_test.size();
//	vector<std::string> labels = readClassNames();
//	float count_error{};
//	char file_name[50];
//	//关闭opencv输出的一大堆调试信息
//	cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
//	//environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
//	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PARSeq");
//	Ort::SessionOptions session_options;
//	// 使用1个线程执行op,若想提升速度，增加线程数
//	session_options.SetIntraOpNumThreads(5);
//	//CUDA加速开启(由于onnxruntime的版本太高，无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
//#ifdef _CUDA_
//
//	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
//#endif
//
//	// ORT_ENABLE_ALL: 启用所有可能的优化
//	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//
//	//load  model and creat session
//
//#ifdef _WIN32
//	const wchar_t* model_path = L"F:\ww\OCR\\onnxruntime_inference_PARSeq\\onnxruntime_inference_PARSeq\\workspace\\parseq.onnx";
//#else
//	const char* model_path = "H:\\0824OCR\\onnxruntime_inference_PARSeq\\onnxruntime_inference_PARSeq\\workspace\\parseq.onnx";
//#endif
//
//	printf("Using Onnxruntime C++ API\n");
//	Session* session = nullptr;
//	session = new Session(env, model_path, session_options);
//	//Ort::Session session(env, model_path, session_options);
//	// print model input layer (node names, types, shape etc.)
//	Ort::AllocatorWithDefaultOptions allocator;
//
//	//model info
//	// 获得模型又多少个输入和输出，一般是指对应网络层的数目
//	// 一般输入只有图像的话input_nodes为1
//	size_t num_input_nodes = session.GetInputCount();
//	// 如果是多输出网络，就会是对应输出的数目
//	size_t num_output_nodes = session.GetOutputCount();
//	printf("Number of inputs = %zu\n", num_input_nodes);
//	printf("Number of output = %zu\n", num_output_nodes);
//	//获取输入name;
//	//GetInputName和GetOutputName函数由于版本更新无法使用
//	const char* input_name;
//	// 	AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
//	// 	input_name = input_name_Ptr.get();
//	input_name = session.GetInputName(0, allocator);
//	std::cout << "input_name:" << input_name << std::endl;
//
//	//获取输出name
//	const char* output_name;
//	//AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(0, allocator);
//	output_name = session.GetOutputName(0, allocator);
//	std::cout << "output_name:" << output_name << std::endl;
//	//自动获取维度数量
//	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
//	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
//	std::cout << "input_dims:" << input_dims[0] << std::endl;
//	std::cout << "output_dims:" << output_dims[0] << std::endl;
//
//
//	std::vector<const char*> input_names{ input_name };
//	std::vector<const char*> output_names = { output_name };
//
//	std::vector<const char*> input_node_names = { "input.1" };
//	std::vector<const char*> output_node_names = { "1581" };
//
//	//加载图片
//
//
//	string Str_num{};
//	string ImgPath{};
//
//	clock_t startTime, endTime;
//	/*************************GPU预热*******************************/
//// 	Mat blob_PreHeat = Blob_PreHeat(Img_PreHeat);
//// 	//创建输入tensor
//// 	auto memory_info_PreHeat = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//// 	std::vector<Ort::Value> input_tensors_PreHeat;
//// 
//// 	input_tensors_PreHeat.emplace_back(Ort::Value::CreateTensor<float>(memory_info_PreHeat, blob_PreHeat.ptr<float>(), blob_PreHeat.total(), input_dims.data(), input_dims.size()));
//// 	startTime = clock();
//// 	//推理(score model & input tensor, get back output tensor)
//// 	auto output_tensors_PreHeat = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors_PreHeat.data(), input_names.size(), output_node_names.data(), output_names.size());
//// 	endTime = clock();
//// 	std::cout << (endTime - startTime) / 1000 << std::endl;
//	/*************************GPU预热*******************************/
//	vector<string> Error_index;
//	vector<string> label_copy;
//	for (int i = 0; i < _count_; i++)
//	{
//		startTime = clock();
//		Mat img = imread(src_test[i]);
//
//		Mat det1, det2;
//		resize(img, det1, Size(IMG_H, IMG_W));
//
//		det1.convertTo(det1, CV_32FC3);
//
//		PreProcess(det1, det2);         //标准化处理
//		// 	cout << "行数" << det2.rows << endl;
//		// 	cout << "列数" << det2.cols << endl;
//		// 	cout << "通道数" << det2.channels() << endl;
//		// 	cout << "维数" << det2.dims << endl;
//			//cout << det2;
//			//imshow("Image ", img);
//		Mat blob = dnn::blobFromImage(det2, 1., Size(IMG_H, IMG_W), Scalar(0, 0, 0), true, false);
//
//		std::cout << blob.channels() << " " << blob.cols << " " << blob.rows << blob.dims << std::endl;
//
//		//printf("Load success!\n");
//	// 	cout << "行数" << blob.rows << endl;
//	// 	cout << "列数" << blob.cols << endl;
//	// 	cout << "通道数" << blob.channels() << endl;
//	// 	cout << "维数" << blob.dims << endl;
//		//创建输入tensor
//		// 
//		// 
//		//表示的是p_data是存储在CPU还是GPU（CUDA）上。
//		//这里我们用CPU来存储输入的Tensor数据即可，因为代码会比较简练：
//	// 	如果是GPU存储，则需要调用CUDA的API，稍微繁琐一点。
//	// 
//	//  即使这里是CPU也不影响我们模型在GPU上跑推理的。
//		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//		std::vector<Ort::Value> input_tensors;
//		std::cout << input_dims.data() << "  " << input_dims.size() << "  " << blob.total() << std::endl;
//		input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
//		/*cout << int(input_dims.size()) << endl;*/
//
//		startTime = clock();
//
//		//推理(score model & input tensor, get back output tensor)
//		auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_names.size());
//		endTime = clock();
//
//		assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
//
//		//除了第一个节点外，其他参数与原网络对应不上程序就会无法执行
//		//第二个参数代表输入节点的名称集合
//		//第四个参数1代表输入层的数目
//		//第五个参数代表输出节点的名称集合
//		//最后一个参数代表输出节点的数目
//		//获取输出(Get pointer to output tensor float values)
//		//float* floatarr = output_tensors[0].GetTensorMutableData<float>();     // 也可以使用output_tensors.front(); 获取list中的第一个元素变量  list.pop_front(); 删除list中的第一个位置的元素																   
//
//		Ort::Value& output_tensor = output_tensors[0];
//
//		Ort::TensorTypeAndShapeInfo tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
//		std::vector<int64_t> output_shape(tensor_info.GetShape());
//		size_t num_dims = tensor_info.GetDimensionsCount();
//		//"Output Shape: 1 26 95"
//		//std::vector<int64_t> output_shape(num_dims);
//		//tensor_info.GetDimensions(output_shape.data(), num_dims);
//		std::vector<std::vector<float>> data_char{};//用于存储解析出的变量，size为26×95
//		std::vector<float> row{};//用于存储解析出的变量，size为1×95
//
//		std::vector<std::vector<float>> probabilities{};//用于存储概率值，size为26×95
//		std::vector<float> probability{};//用于存储概率值，size为1×95
//		std::vector<float> sum{};//用于存储26个概率值的和。size为1×26
//		float max_value{};
//		float sum_tem{};//用于存储临时
//		if (output_shape.size() == 3 && output_shape[0] == 1 && output_shape[1] == 26 && output_shape[2] == 95)
//		{
//			// 获取数据指针
//			float* data = output_tensors[0].GetTensorMutableData<float>();
//			//Mat newarr = Mat_<double>(26, TOTAL_CATEGORY); //定义一个1*95的矩阵
//			// 输出数据
//			for (int i = 0; i < output_shape[1]; i++)
//			{
//				for (int j = 0; j < output_shape[2]; j++)
//				{
//					int index = i * output_shape[2] + j;
//					//newarr.at<double>(i, j) = data[index];
//					row.push_back(data[index]);
//				}
//				data_char.push_back(row);
//				row.clear();
//			}
//		}
//		else
//		{
//			std::cout << "Invalid shape!" << std::endl;
//		}
//
//		//处理得到最后的类别
//		for (int i = 0; i < data_char.size(); i++)
//		{
//			max_value = *std::max_element(data_char[i].begin(), data_char[i].end());
//			for (int j = 0; j < data_char[i].size(); j++) //矩阵列数循环
//			{
//				probability.push_back(std::exp(data_char[i][j] - max_value));
//				sum_tem += probability[j];
//			}
//			sum.push_back(sum_tem);
//			probabilities.push_back(probability);
//			probability.clear();
//		}
//
//		for (int i = 0; i < probabilities.size(); i++)
//		{
//			for (int j = 0; j < probabilities[0].size(); j++)
//			{
//				probabilities[i][j] /= sum[i];
//			}
//
//		}
//		std::cout << "字符类别为：\n";
//		for (int i = 0; i < probabilities.size(); i++)
//		{
//			auto max_prob = std::max_element(probabilities[i].begin(), probabilities[i].end());
//			int index_label = max_prob - probabilities[i].begin();
//			if (index_label == 0)//遇到截至符号，就break
//				break;
//			std::cout << labels.at(index_label).c_str();
//			std::cout << "  " << index_label << std::endl;
//		}
//		endTime = clock();
//		std::cout << "time:" << (endTime - startTime) << std::endl;
//
//	}
//	system("pause");
//	return 0;
//}
