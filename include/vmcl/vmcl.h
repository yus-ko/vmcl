#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
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
#include <nav_msgs/Path.h>

#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>

#include <potbot_lib/utility_ros.h>
#include <potbot_lib/filter.h>
#include <vmcl/particle.h>

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
		std::string frame_id = "";
		potbot_lib::Pose pose;

		geometry_msgs::PoseStamped to_msg() const
		{
            geometry_msgs::PoseStamped marker_msg;
			marker_msg.pose = potbot_lib::utility::get_pose(pose);
			marker_msg.header.frame_id = frame_id;
			return marker_msg;
        }
	};
	
	class VMCLNode
	{
		private:

			tf2_ros::Buffer* tf_buffer_;

			std::string source_frame_ = "map";
			std::string frame_id_camera_ = "camera_link";
			std::string frame_id_robot_ = "base_footprint";
			std::string frame_id_odom_ = "odom";

			ros::NodeHandle nh_sub_;
			ros::Subscriber sub_odom_, sub_inipose_;
			message_filters::Subscriber<sensor_msgs::Image> sub_rgb_;
			message_filters::Subscriber<sensor_msgs::Image> sub_depth_;
			message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;
			ros::Publisher pub_estimate_pose_, pub_particles_, pub_odometry_, pub_observed_marker_, pub_observed_marker_img_;
			tf2_ros::TransformBroadcaster dynamic_br_;

			bool using_particle_filter_ = true;
			Particle* particle_ = nullptr;

			potbot_lib::filter::MoveMeanPose* pose_filter_ = nullptr;
			// potbot_lib::filter::LowPassPose* pose_filter_ = nullptr;

			std::vector<int> observed_marker_ids_pre_;
			double correct_distance_ = 2.0;
			double depth_scaling_ = 1.0;
			double resampling_period_ = 1.0;
			geometry_msgs::Pose pose_difference_, pose_target_;

			typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
			boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

			nav_msgs::Odometry encoder_odometry_;
			std::vector<Marker> observed_markers_;
			potbot_lib::Pose estimated_pose_;

			dynamic_reconfigure::Server<vmcl::VMCLConfig> *dsrv_;
			potbot_lib::Point debug_eular_;
			int move_mean_window_num_ = 10;
			double low_pass_coefficient_ = 0.5;

			void reconfigureCallback(const vmcl::VMCLConfig& param, uint32_t level); 

			void imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg);
			void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
			void iniposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);

			bool getMarkerCoords(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg, std::vector<Marker>& markers);
			Marker getMarkerTruth(int id);
			potbot_lib::Pose getRobotFromMarker(const std::vector<Marker>& markers);
			void publishMarker(const std::vector<Marker>& markers);

			void fixOdomPose();

		public:
			VMCLNode(tf2_ros::Buffer* tf);
			~VMCLNode();
	};
}

#endif