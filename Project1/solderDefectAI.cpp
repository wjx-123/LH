#include "solderDefectAI.h"



OnnxModelAI::OnnxModelAI(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel"),session(env, stringToWstring(model_path).c_str(), Ort::SessionOptions{ nullptr }) {
    num_input_nodes = session.GetInputCount();
    num_output_nodes = session.GetOutputCount();

    // 获取输入输出的名称和维度
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    input_names = { input_name };
    output_names = { output_name };
}

std::wstring OnnxModelAI::stringToWstring(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}

void OnnxModelAI::preprocess(cv::Mat& imgSrc, std::vector<float>& inputTensorValues) {
    // 图像预处理逻辑...
    center_resize(imgSrc);
    cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
    imgSrc.convertTo(imgSrc, CV_32F, 1.0 / 255);  //divided by 255转float
    cv::Mat channels[3]; //分离通道进行HWC->CHW
    cv::split(imgSrc, channels);
    float mean[] = { 0.485f, 0.456f, 0.406f };	//
    float std_val[] = { 0.229f, 0.224f, 0.225f };
    for (int i = 0; i < imgSrc.channels(); i++)	//标准化ImageNet
    {
        channels[i] -= mean[i];  // mean均值
        channels[i] /= std_val[i];   // std方差
    }
    for (int i = 0; i < imgSrc.channels(); i++)  //HWC->CHW
    {
        std::vector<float> data = std::vector<float>(channels[i].reshape(1, imgSrc.cols * imgSrc.rows));
        inputTensorValues.insert(inputTensorValues.end(), data.begin(), data.end());
    }
}

void OnnxModelAI::center_resize(cv::Mat& src)
{
    int width = src.cols;
    int height = src.rows;
    cv::Point center(width / 2, height / 2); // 指定的中心点 
    cv::Size size(300, 300); // 裁剪出的图片大小 
    getRectSubPix(src, size, center, src); // 裁剪出目标图片
}

std::pair<int, float> OnnxModelAI::predict(cv::Mat& imgSrc) {
    std::vector<float> inputTensorValues;
    preprocess(imgSrc, inputTensorValues);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), input_dims.data(), input_dims.size()));

    auto outputTensor = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), inputTensors.data(), 1, output_names.data(), 1);

    // 后处理逻辑，例如提取预测类别和置信度
    const int num_classes = 4;
    float* prob = outputTensor[0].GetTensorMutableData<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    float confidence = prob[predict_label];

    return { predict_label, confidence };
}