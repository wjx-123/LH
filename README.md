# LH
AIAlgorithmToDetectSolderJointDefects.h 	焊点缺陷检测的智能方法，但不规范，重写后见solderDefectAI
biSenetOpenvion.h 		双边语义分割bisenetv2的部署
CameraCalibration.h 	相机模组自动标定
circularDectection.h		环形图片展平的缺陷检测
colorMethod.h		重写的颜色方法
edline.h			edline直线检测
eliminateYoloBackground.h	贴片内框的传统检测方法
fov_puzzle.h		fov规划后检测拼图 包含矫正偏移、角度等
fovRoutePlan.h		路径规划
fovTest.h 			路劲规划测试
halconMatch.h		halcon匹配工具类以及拼版相关函数
onnx.h			使用yolo+unet的部署 速度慢
onnx2.h			传统检测贴片内框方法+生产者消费者模型
onnxInit.h			初始onnx
patchDetection.h		贴片的传统检测方法 有无、偏移等
solderDefectAI.h		重写AIAlgorithmToDetectSolderJointDefects.h 
splicingBoard.h		检测两张图片的微小差距 模板匹配、直线筛选法 已废弃
uNetOnnx.h		unet部署方法