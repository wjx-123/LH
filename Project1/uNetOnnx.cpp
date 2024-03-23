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
        // 加载模型
        Ort::Session session = loadModel(L"model_v2_sim.onnx");

        // 读取输入图像
        cv::Mat input_image = img;
        if (input_image.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
            return res;
        }
        
        // 执行推理
        cv::Mat segmentation_result = performInference(session, input_image);

        // 显示结果
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

    // 检查是否支持GPU
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
        //原始图片大小
        int org_width = img.cols;
        int org_height = img.rows;

        //计算缩放比例
        double scale = std::min(1.0, std::min((double)512 / org_width, (double)512 / org_height));

        //缩放图片，并保持长宽比
        cv::Mat resized_img;
        resize(img, resized_img, cv::Size(), scale, scale);

        //计算灰度填充的大小
        int pad_x = (512 - resized_img.cols) / 2;
        int pad_y = (512 - resized_img.rows) / 2;

        //加上灰色填充
        cv::Mat padded_img;
        copyMakeBorder(resized_img, padded_img, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

        //图片通道转换和归一化
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

        cv::Mat predMatColor = cv::Mat::zeros(H, W, CV_8UC3); // 创建一个三通道的彩色图像
        // 定义三个类别的颜色
        cv::Vec3b colorBackground(0, 0, 0); // 背景 - 黑色
        cv::Vec3b colorClass1(0, 255, 0);    // 类别1 - 绿色
        cv::Vec3b colorClass2(0, 0, 255);    // 类别2 - 蓝色

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float bg = output.ptr<float>(0, 0, h)[w];
                float c1 = output.ptr<float>(0, 1, h)[w];
                float c2 = output.ptr<float>(0, 2, h)[w];
                //float c3 = output.ptr<float>(0, 3, h)[w];
                if (bg >= c1 && bg >= c2) {//背景
                    //predMat.at<float>(h, w) = 0;
                    predMatColor.at<cv::Vec3b>(h, w) = colorBackground;
                }
                else if (c1 >= c2 ) {//类别1
                    //predMat.at<float>(h, w) = 150;
                    predMatColor.at<cv::Vec3b>(h, w) = colorClass1;
                }
                else if (c2 > c1 ) {//类别2
                    //predMat.at<float>(h, w) = 255;
                    predMatColor.at<cv::Vec3b>(h, w) = colorClass2;
                }
                else {//类别3
                    //predMat.at<float>(h, w) = 0;
                }
            }
        }
        //去除灰色填充并还原原始大小
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
    //// 检测绿色和红色区域
    //cv::Mat maskGreen, maskRed;
    //cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    //cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    //// 找到绿色区域的轮廓
    //std::vector<std::vector<cv::Point>> contoursGreen;
    //cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// 找到红色区域的轮廓
    //std::vector<std::vector<cv::Point>> contoursRed;
    //cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// 存储所有矩形的向量
    //std::vector<cv::Rect2f> boxes;

    //// 对于绿色区域，假设只有一个区域
    //if (!contoursGreen.empty()) {
    //    boxes.push_back(cv::boundingRect(contoursGreen[0]));
    //}

    //// 对于每个红色区域，计算外接矩形并添加到向量中
    //for (const auto& contour : contoursRed) {
    //    boxes.push_back(cv::boundingRect(contour));
    //}

    //// 绘制所有矩形
    //for (const auto& box : boxes) {
    //    cv::rectangle(uNetImg, box, cv::Scalar(255, 0, 0), 2);
    //}

    //// 显示图像
    //cv::imshow("Image with Bounding Boxes", uNetImg);
    //cv::waitKey(0);

    // 检测绿色和红色区域
    cv::Mat maskGreen, maskRed;
    cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    // 找到绿色区域的轮廓
    std::vector<std::vector<cv::Point>> contoursGreen;
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 找到红色区域的轮廓
    std::vector<std::vector<cv::Point>> contoursRed;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 存储所有旋转矩形的向量
    std::vector<cv::RotatedRect> boxes;

    // 对于绿色区域，假设只有一个区域
    if (!contoursGreen.empty()) {
        boxes.push_back(cv::minAreaRect(contoursGreen[0]));
    }

    // 对于每个红色区域，计算旋转外接矩形并添加到向量中
    for (const auto& contour : contoursRed) {
        boxes.push_back(cv::minAreaRect(contour));
    }

    return boxes;
}


std::vector<cv::Rect> uNetOnnx::getMultipodPackRect(cv::Mat uNetImg)
{
    // 检测绿色和红色区域
    cv::Mat maskGreen, maskRed;
    cv::inRange(uNetImg, cv::Scalar(0, 128, 0), cv::Scalar(0, 255, 0), maskGreen);
    cv::inRange(uNetImg, cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 255), maskRed);

    // 找到绿色区域的轮廓
    std::vector<std::vector<cv::Point>> contoursGreen;
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 找到红色区域的轮廓
    std::vector<std::vector<cv::Point>> contoursRed;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 存储所有矩形的向量
    std::vector<cv::Rect> boxes;

    // 对于绿色区域，假设只有一个区域
    if (!contoursGreen.empty()) {
        boxes.push_back(cv::boundingRect(contoursGreen[0]));
    }

    // 对于每个红色区域，计算外接矩形并添加到向量中
    for (const auto& contour : contoursRed) {
        cv::Rect temp = cv::boundingRect(contour);
        if (temp.height * temp.width < 2500)
        {
            continue;
        }
        boxes.push_back(cv::boundingRect(contour));
    }

    // 绘制所有矩形
    for (const auto& box : boxes) {
        cv::rectangle(uNetImg, box, cv::Scalar(255, 0, 0), 2);
    }
    ////// 显示图像
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

    // 获取输入和输出节点信息
    char* input_node_name = session.GetInputName(0, allocator);
    char* output_node_name = session.GetOutputName(0, allocator);

    // 预处理输入图像
    cv::Mat img;
    cv::cvtColor(input_image, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(2048, 1024));  // 与模型输入大小匹配
    img.convertTo(img, CV_32F, 1.0 / 255);       // 归一化

    // 创建输入张量
    size_t img_size = img.total() * img.elemSize();
    std::vector<float> img_data(img_size);
    std::memcpy(img_data.data(), img.data, img_size);
    std::vector<int64_t> input_tensor_shape = {1, 3, 1024, 2048};  // 模型期望的输入形状

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        img_data.data(),
        img_data.size(),
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    // 运行模型
    std::vector<const char*> input_names{ input_node_name };
    std::vector<const char*> output_names{ output_node_name };
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // 释放资源
    allocator.Free(input_node_name);
    allocator.Free(output_node_name);

    // 处理输出张量
    auto& output_tensor = output_tensors.front();
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> output_dims(output_shape.begin(), output_shape.end());
    cv::Mat result(output_dims[2], output_dims[3], CV_32SC1, output_tensor.GetTensorMutableData<int>());

    // 将输出张量转换为图像 (这里需要您的mapToColors函数将类别映射到颜色)
    cv::Mat color_segmentation = mapToColors(result); // 您需要定义这个函数

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

    // 获取输入和输出节点名称
    char* input_name1 = session.GetInputName(0, allocator);
    char* output_name = session.GetOutputName(0, allocator);
    std::vector<const char*> input_name{ input_name1 };
    std::vector<const char*> preds{ output_name };

    // 运行模型
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_name.data(), &input_tensor, 1, preds.data(), 1);

    // 释放资源
    allocator.Free(input_name1);
    allocator.Free(output_name);

    return output_tensors;
}
