#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>         //画像変換のヘッダ
#include <sensor_msgs/Image.h>           //センサーデータ形式ヘッダ
#include <sensor_msgs/image_encodings.h> //エンコードのためのヘッダ
#include <sensor_msgs/CameraInfo.h>      //camera_infoを獲得するためのヘッダー
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <random>
#include <geometry_msgs/PoseStamped.h> //tf
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <opencv2/aruco/charuco.hpp> //マーカー検出
#include <nav_msgs/Path.h>           //経路情報を記録する

#include <geometry_msgs/Twist.h> //ロボットの指令値(速度)用ヘッダー
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>

#include <potbot_lib/utility_ros.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

#include <dynamic_reconfigure/server.h>
#include <vmcl/VMCLConfig.h>

namespace vmcl
{
	struct Marker
	{
		int id = 0;
		std::string frame_id;
		potbot_lib::Pose pose;
	};

	class VMCLNode
	{
		private:

			tf2_ros::Buffer* tf_buffer_;

			std::string source_frame_ = "map"; //mapフレーム
			std::string frame_id_camera_ = "camera_link"; //mapフレーム

			ros::NodeHandle nh_sub_;
			ros::Subscriber sub_odom_;
			message_filters::Subscriber<sensor_msgs::Image> sub_rgb_;
			message_filters::Subscriber<sensor_msgs::Image> sub_depth_;
			ros::Publisher pub_estimate_odometry_, pub_particles_, pub_odometry_, pub_observed_marker_;

			geometry_msgs::Twist velocity_command_;                //指令速度

			std::vector<int> observed_marker_ids_pre_;
			double correct_distance_ = 2.0;
			geometry_msgs::Pose pose_diffetence_;

			double particle_num_ = 100;//パーティクル個数

			std::vector<potbot_lib::Pose> particles_;	//パーティクルの位置
			std::vector<double> particle_weight_;//各パーティクルに対する重み
			std::vector<std::vector<double>> noise_;
			std::vector<std::vector<double>> noise_params_;

			std::vector<double> breez_;//ノイズ付きパーティクル速度
			std::vector<double> greed_;//ノイズ付きパーティクル

			// ApproximateTimeポリシーの定義
			typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
			
			// 同期器の定義
			boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

			nav_msgs::Odometry encoder_odometry_;

			dynamic_reconfigure::Server<vmcl::VMCLConfig> *dsrv_;

			void reconfigureCallback(const vmcl::VMCLConfig& param, uint32_t level); 

			void imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg,const sensor_msgs::Image::ConstPtr& depth_msg);
			void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);

			bool getMarkerCoords(cv::Mat img_src, cv::Mat img_depth, std::vector<Marker>& markers);
			Marker getMarkerTruth(int id);
			void initParticles();
			void updateParticles();

		public:
			VMCLNode(tf2_ros::Buffer* tf);
			~VMCLNode();
	};
}

#endif