#pragma once
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class biSeNetOpenvino {
public:
	biSeNetOpenvino();
	~biSeNetOpenvino();

	void test();
private:
	void get_image(std::string, std::vector<unsigned long>, float*);
	std::vector<std::vector<uint8_t>> get_color_map();
	void save_predict(std::string, int*,
		std::vector<unsigned long>, std::vector<unsigned long>);
	void print_infos();
	void inference();
	void test_speed();

	cv::Mat resizeImgOfSave(cv::Mat iniImg, cv::Mat saveImg);


	//std::string mdpth("../output_v2/model_v2_city.xml");
	//std::string device("CPU"); // GNA does not support argmax, my cpu does not has integrated gpu
	//std::string impth("../../example.png");
	//std::string savepth("./res.jpg");
	std::string mdpth = "E:\\visualworkplace\\Project1\\Project1\\model_v2_sim.onnx";
	std::string device = "GPU";
	std::string impth = "C:\\Users\\LENOVO\\Pictures\\smt\\2499.jpg";
	std::string savepth = "C:\\Users\\LENOVO\\Pictures\\smt\\fov1_res.jpg";
};