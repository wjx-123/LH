#include "AIAlgorithmToDetectSolderJointDefects.h"

void center_resize(cv::Mat& src)//���Ĳü�����2
{
	int width = src.cols;
	int height = src.rows;
	cv::Point center(width / 2, height / 2); // ָ�������ĵ� 
	cv::Size size(300, 300); // �ü�����ͼƬ��С 
	getRectSubPix(src, size, center, src); // �ü���Ŀ��ͼƬ
}

void SegmentAIONNX(cv::Mat& imgSrc, int width, int height)
{
	//ģ����Ϣ /
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	const wchar_t* model_path = L"acc-76.onnx";

	Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;
	size_t num_input_nodes = session.GetInputCount();	//batchsize
	size_t num_output_nodes = session.GetOutputCount();
	const char* input_name = session.GetInputName(0, allocator);
	const char* output_name = session.GetOutputName(0, allocator);
	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();	//�������ά��
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<const char*> input_names{ input_name };
	std::vector<const char*> output_names = { output_name };
	clock_t startTime, endTime;	//��������ʱ��
	//startTime = clock();
	//���봦��//
	cv::Mat imgBGR = imgSrc;	//����ͼƬԤ����
	cv::Mat imgRGBresize;
	center_resize(imgBGR);
	//resize(imgBGR, imgBGRresize, Size(input_dims[3], input_dims[2]), InterpolationFlags::INTER_CUBIC);

	cvtColor(imgBGR, imgRGBresize, cv::COLOR_BGR2RGB);	//smpδתRGB
	cv::Mat resize_img;
	imgRGBresize.convertTo(resize_img, CV_32F, 1.0 / 255);  //divided by 255תfloat
	cv::Mat channels[3]; //����ͨ������HWC->CHW
	cv::split(resize_img, channels);
	std::vector<float> inputTensorValues;
	float mean[] = { 0.485f, 0.456f, 0.406f };	//
	float std_val[] = { 0.229f, 0.224f, 0.225f };
	for (int i = 0; i < resize_img.channels(); i++)	//��׼��ImageNet
	{
		channels[i] -= mean[i];  // mean��ֵ
		channels[i] /= std_val[i];   // std����
	}
	for (int i = 0; i < resize_img.channels(); i++)  //HWC->CHW
	{
		std::vector<float> data = std::vector<float>(channels[i].reshape(1, resize_img.cols * resize_img.rows));
		inputTensorValues.insert(inputTensorValues.end(), data.begin(), data.end());
	}
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), input_dims.data(), input_dims.size()));
	startTime = clock();
	auto outputTensor = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), inputTensors.data(), 1, output_names.data(), 1);   // ��ʼ����
	//endTime = clock();
	//��ӡģ����Ϣ /
		//printf("Using Onnxruntime C++ API\n");
		//printf("Number of inputs = %zu\n", num_input_nodes);
		//printf("Number of output = %zu\n", num_output_nodes);
		//std::cout << "input_name:" << input_name << std::endl;
		//std::cout << "output_name: " << output_name << std::endl;
		//std::cout << "input_dims:" << input_dims[0] << input_dims[1] << input_dims[2] << input_dims[3] << std::endl;
		//std::cout << "output_dims:" << output_dims[0] << output_dims[1] << output_dims[2] << output_dims[3] << std::endl;
		//std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	//�������//
	const int num_classes = 4;
	float* prob = outputTensor[0].GetTensorMutableData<float>(); //outtensor�׵�ַ
	int predict_label = std::max_element(prob, prob + num_classes) - prob;  // ȷ��Ԥ�������±�

	float confidence = prob[predict_label];    // ���Ԥ��ֵ�����Ŷ�
	printf("confidence = %f, label = %d\n", confidence, predict_label);
	endTime = clock();//��ʱ����
	std::cout << "total����ʱ��: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
	//vector< unsigned char >results(512 * 512);
	//for (int i = 0; i < 512 * 512; i++)
	//{
	//	if (mask_ptr[i] >= 0.5)
	//	{
	//		results[i] = 0;
	//	}
	//	else
	//	{
	//		results[i] = 255;
	//	}
	//}
	//unsigned char* ptr = &results[0];
	//Mat mask = Mat(output_dims[2], output_dims[3], CV_8U, ptr);
	//resize(mask, imgSrc, Size(imgBGR.cols, imgBGR.rows));
	//ԭͼչʾ�ָ���//
	//cvtColor(imgSrc, imgSrc, COLOR_GRAY2BGR);
	//Mat imgAdd;
	//addWeighted(imgBGR, 1, imgSrc, 0.3, 0, imgAdd);
}