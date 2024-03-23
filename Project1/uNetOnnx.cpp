#include "uNetOnnx.h"

uNetOnnx::uNetOnnx()
{
}

uNetOnnx::~uNetOnnx()
{
}

std::vector<cv::Rect> uNetOnnx::getDnnClassesRect(cv::Mat img, cv::dnn::Net uNetModel)
{
    cv::Mat uNetImg = dnnGetClasses(img, uNetModel);//dnn
    std::vector<cv::Rect> res = getMultipodPackRect(uNetImg);
    try {
        // ����ģ��
        Ort::Session session = loadModel(L"model_v2_sim.onnx");

        // ��ȡ����ͼ��
        cv::Mat input_image = img;
        if (input_image.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
            return res;
        }
        
        // ִ������
        cv::Mat segmentation_result = performInference(session, input_image);

        // ��ʾ���
        cv::imshow("Segmentation Result", segmentation_result);
        cv::waitKey(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return res;
    }

    return res;

    //return res;
}

cv::dnn::Net uNetOnnx::loadUnetModel(std::string& modelFile)
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelFile);

    // ����Ƿ�֧��GPU
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "GPU found, setting CUDA backend and target" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else {
        std::cout << "GPU not found, using CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    return net;
}

cv::Mat uNetOnnx::dnnGetClasses(cv::Mat img, cv::dnn::Net uNetModel)
{
    cv::Mat roi_img;
    try {
        //ԭʼͼƬ��С
        int org_width = img.cols;
        int org_height = img.rows;

        //�������ű���
        double scale = std::min(1.0, std::min((double)512 / org_width, (double)512 / org_height));

        //����ͼƬ�������ֳ����
        cv::Mat resized_img;
        resize(img, resized_img, cv::Size(), scale, scale);

        //����Ҷ����Ĵ�С
        int pad_x = (512 - resized_img.cols) / 2;
        int pad_y = (512 - resized_img.rows) / 2;

        //���ϻ�ɫ���
        cv::Mat padded_img;
        copyMakeBorder(resized_img, padded_img, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

        //ͼƬͨ��ת���͹�һ��
        padded_img.convertTo(padded_img, CV_32FC3, 1.f / 255.f);

        cv::resize(padded_img, padded_img, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
        cv::Mat inputBolb = cv::dnn::blobFromImage(padded_img);

        uNetModel.setInput(inputBolb);
        cv::Mat output = uNetModel.forward();

        int N = inputBolb.size[0];
        int C = inputBolb.size[1];
        int H = inputBolb.size[2];
        int W = inputBolb.size[3];
        //cv::Mat predMat = cv::Mat::zeros(512, 512, CV_32F);

        cv::Mat predMatColor = cv::Mat::zeros(H, W, CV_8UC3); // ����һ����ͨ���Ĳ�ɫͼ��
        // ��������������ɫ
        cv::Vec3b colorBackground(0, 0, 0); // ���� - ��ɫ
        cv::Vec3b colorClass1(0, 255, 0);    // ���1 - ��ɫ
        cv::Vec3b colorClass2(0, 0, 255);    // ���2 - ��ɫ

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float bg = output.ptr<float>(0, 0, h)[w];
                float c1 = output.ptr<float>(0, 1, h)[w];
                float c2 = output.ptr<float>(0, 2, h)[w];
                //float c3 = output.ptr<float>(0, 3, h)[w];
                if (bg >= c1 && bg >= c2) {//����
                    //predMat.at<float>(h, w) = 0;
                    predMatColor.at<cv::Vec3b>(h, w) = colorBackground;
                }
                else if (c1 >= c2 ) {//���1
                    //predMat.at<float>(h, w) = 150;
                    predMatColor.at<cv::Vec3b>(h, w) = colorClass1;
                }
                else if (c2 > c1 ) {//���2
                    //predMat.at<float>(h, w) = 255;
                    predMatColor.at<cv::Vec3b>(h, w) = colorClass2;
                }
                else {//���3
                    //predMat.at<float>(h, w) = 0;
                }
            }
        }
        //ȥ����ɫ��䲢��ԭԭʼ��С
        cv::Rect roi(pad_x, pad_y, org_width * scale, org_height * scale);
        roi_img = predMatColor(roi);
        cv::resize(roi_img, roi_img, cv::Size(org_width, org_height));
    }
    catch (cv::Exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "wrong" << std::endl;
    }
    return roi_img;
}

std::vector<cv::RotatedRect> uNetOnnx::getMultipodPackRoatedRect(cv::Mat uNetImg)
{
    //// �����ɫ�ͺ�ɫ����
    //cv::Mat maskGreen, maskRed;
    //cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    //cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    //// �ҵ���ɫ���������
    //std::vector<std::vector<cv::Point>> contoursGreen;
    //cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// �ҵ���ɫ���������
    //std::vector<std::vector<cv::Point>> contoursRed;
    //cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// �洢���о��ε�����
    //std::vector<cv::Rect2f> boxes;

    //// ������ɫ���򣬼���ֻ��һ������
    //if (!contoursGreen.empty()) {
    //    boxes.push_back(cv::boundingRect(contoursGreen[0]));
    //}

    //// ����ÿ����ɫ���򣬼�����Ӿ��β���ӵ�������
    //for (const auto& contour : contoursRed) {
    //    boxes.push_back(cv::boundingRect(contour));
    //}

    //// �������о���
    //for (const auto& box : boxes) {
    //    cv::rectangle(uNetImg, box, cv::Scalar(255, 0, 0), 2);
    //}

    //// ��ʾͼ��
    //cv::imshow("Image with Bounding Boxes", uNetImg);
    //cv::waitKey(0);

    // �����ɫ�ͺ�ɫ����
    cv::Mat maskGreen, maskRed;
    cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    // �ҵ���ɫ���������
    std::vector<std::vector<cv::Point>> contoursGreen;
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // �ҵ���ɫ���������
    std::vector<std::vector<cv::Point>> contoursRed;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // �洢������ת���ε�����
    std::vector<cv::RotatedRect> boxes;

    // ������ɫ���򣬼���ֻ��һ������
    if (!contoursGreen.empty()) {
        boxes.push_back(cv::minAreaRect(contoursGreen[0]));
    }

    // ����ÿ����ɫ���򣬼�����ת��Ӿ��β���ӵ�������
    for (const auto& contour : contoursRed) {
        boxes.push_back(cv::minAreaRect(contour));
    }

    return boxes;
}


std::vector<cv::Rect> uNetOnnx::getMultipodPackRect(cv::Mat uNetImg)
{
    // �����ɫ�ͺ�ɫ����
    cv::Mat maskGreen, maskRed;
    cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    // �ҵ���ɫ���������
    std::vector<std::vector<cv::Point>> contoursGreen;
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // �ҵ���ɫ���������
    std::vector<std::vector<cv::Point>> contoursRed;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // �洢���о��ε�����
    std::vector<cv::Rect> boxes;

    // ������ɫ���򣬼���ֻ��һ������
    if (!contoursGreen.empty()) {
        boxes.push_back(cv::boundingRect(contoursGreen[0]));
    }

    // ����ÿ����ɫ���򣬼�����Ӿ��β���ӵ�������
    for (const auto& contour : contoursRed) {
        cv::Rect temp = cv::boundingRect(contour);
        if (temp.height * temp.width < 2500)
        {
            continue;
        }
        boxes.push_back(cv::boundingRect(contour));
    }

    // �������о���
    for (const auto& box : boxes) {
        cv::rectangle(uNetImg, box, cv::Scalar(255, 0, 0), 2);
    }
    ////// ��ʾͼ��
    //cv::imshow("Image with Bounding Boxes", uNetImg);
    //cv::waitKey(0);


    return boxes;
}

Ort::Session uNetOnnx::loadModel(const std::wstring& model_path)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BiSeNet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path.c_str(), session_options);
    return session;
}

cv::Mat uNetOnnx::performInference(Ort::Session& session, const cv::Mat& input_image)
{
     Ort::AllocatorWithDefaultOptions allocator;

    // ��ȡ���������ڵ���Ϣ
    char* input_node_name = session.GetInputName(0, allocator);
    char* output_node_name = session.GetOutputName(0, allocator);

    // Ԥ��������ͼ��
    cv::Mat img;
    cv::cvtColor(input_image, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(2048, 1024));  // ��ģ�������Сƥ��
    img.convertTo(img, CV_32F, 1.0 / 255);       // ��һ��

    // ������������
    size_t img_size = img.total() * img.elemSize();
    std::vector<float> img_data(img_size);
    std::memcpy(img_data.data(), img.data, img_size);
    std::vector<int64_t> input_tensor_shape = {1, 3, 1024, 2048};  // ģ��������������״

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        img_data.data(),
        img_data.size(),
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    // ����ģ��
    std::vector<const char*> input_names{ input_node_name };
    std::vector<const char*> output_names{ output_node_name };
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // �ͷ���Դ
    allocator.Free(input_node_name);
    allocator.Free(output_node_name);

    // �����������
    auto& output_tensor = output_tensors.front();
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> output_dims(output_shape.begin(), output_shape.end());
    cv::Mat result(output_dims[2], output_dims[3], CV_32SC1, output_tensor.GetTensorMutableData<int>());

    // ���������ת��Ϊͼ�� (������Ҫ����mapToColors���������ӳ�䵽��ɫ)
    cv::Mat color_segmentation = mapToColors(result); // ����Ҫ�����������

    return color_segmentation;
}

size_t uNetOnnx::vector_product(const std::vector<int64_t>& vec) {
    size_t prod = 1;
    for (auto& v : vec) prod *= v;
    return prod;
}

cv::Mat uNetOnnx::mapToColors(const cv::Mat& segmentation_result) {
    std::vector<cv::Vec3b> colors = { cv::Vec3b(0, 0, 255), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0) };
    cv::Mat color_segmentation(segmentation_result.size(), CV_8UC3);

    for (int y = 0; y < segmentation_result.rows; y++) {
        for (int x = 0; x < segmentation_result.cols; x++) {
            int label = segmentation_result.at<float>(y, x);
            color_segmentation.at<cv::Vec3b>(y, x) = colors[label];
        }
    }
    return color_segmentation;
}

std::vector<Ort::Value> uNetOnnx::runModel(Ort::Session& session, const Ort::Value& input_tensor) {
    Ort::AllocatorWithDefaultOptions allocator;

    // ��ȡ���������ڵ�����
    char* input_name1 = session.GetInputName(0, allocator);
    char* output_name = session.GetOutputName(0, allocator);
    std::vector<const char*> input_name{ input_name1 };
    std::vector<const char*> preds{ output_name };

    // ����ģ��
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_name.data(), &input_tensor, 1, preds.data(), 1);

    // �ͷ���Դ
    allocator.Free(input_name1);
    allocator.Free(output_name);

    return output_tensors;
}
