#include <vmcl/vmcl.h>

using namespace std;
using namespace cv;

namespace vmcl
{
	VMCLNode::VMCLNode(tf2_ros::Buffer* tf)
	{
		tf_buffer_ = tf;

		ros::NodeHandle pnh("~");

		double m_vv=0, v_vv=0;
		double m_vw=0, v_vw=0;
		double m_wv=0, v_wv=0;
		double m_ww=0, v_ww=0;
		double x=0,y=0,yaw=0;

		pnh.getParam("norm_noise_mean_linear_linear", m_vv);
		pnh.getParam("norm_noise_variance_linear_linear", v_vv);
		pnh.getParam("norm_noise_mean_linear_angular", m_vw);
		pnh.getParam("norm_noise_variance_linear_angular", v_vw);
		pnh.getParam("norm_noise_mean_angular_linear", m_wv);
		pnh.getParam("norm_noise_variance_angular_linear", v_wv);
		pnh.getParam("norm_noise_mean_angular_angular", m_ww);
		pnh.getParam("norm_noise_variance_angular_angular", v_ww);
		pnh.getParam("depth_scaling", depth_scaling_);
		pnh.getParam("frame_id_camera_link", frame_id_camera_);
		pnh.getParam("initial_pose_x", x);
		pnh.getParam("initial_pose_y", y);
		pnh.getParam("initial_pose_yaw", yaw);

		// noise_params_ = {
		// 	{m_vv,v_vv},	//直進で生じる道のり
		// 	{m_vw,v_vw},	//回転で生じる道のり
		// 	{m_wv,v_wv},
		// 	{m_ww,v_ww}
		// };

		pose_difference_ = potbot_lib::utility::get_pose(x,y,0,0,0,yaw);

		//Realsensesの時(roslaunch realsense2_camera rs_camera.launch align_depth:=true)(Depth修正版なのでこっちを使うこと)
		sub_odom_ = nh_sub_.subscribe("odom", 1, &VMCLNode::odomCallback, this);
		sub_inipose_ = nh_sub_.subscribe("initialpose", 1, &VMCLNode::iniposeCallback, this);
		sub_rgb_.subscribe(nh_sub_, "color/image_raw", 1);
		sub_depth_.subscribe(nh_sub_, "depth/image_raw", 1);
		sub_info_.subscribe(nh_sub_, "color/camera_info", 1);

		sync_.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), sub_rgb_, sub_depth_, sub_info_));
		sync_->registerCallback(boost::bind(&VMCLNode::imageCallback, this, _1, _2, _3));

		ros::NodeHandle nhPub;
		pub_estimate_odometry_ = nhPub.advertise<nav_msgs::Odometry>("estimate_odometry",1000);
		pub_particles_ = nhPub.advertise<geometry_msgs::PoseArray>("particles_posearray",1000);
		pub_odometry_ = nhPub.advertise<nav_msgs::Odometry>("debug/encoder_odometry",1000);
		pub_observed_marker_ = nhPub.advertise<visualization_msgs::MarkerArray>("debug/observed_marker",1000);

		//2024-07-31廊下(直進11.0m,速度0.25)直線動作実験
		//rosbag play 2024-07-31-13-10-22.bag//実測値(X:11.400,Y:3.262)
		//rosbag play 2024-07-31-13-34-34.bag//実測値(X:11.050,Y:3.110)
		// LX=11.0,VX=0.25,omegaZ=1.0,THZ=0.20,LY=3.00;

		// initParticles();

		pose_filter_ = new potbot_lib::filter::MoveMeanPose(move_mean_window_num_);
		// pose_filter_ = new potbot_lib::filter::LowPassPose(move_mean_window_num_);
		
		dsrv_ = new dynamic_reconfigure::Server<vmcl::VMCLConfig>(pnh);
		dynamic_reconfigure::Server<vmcl::VMCLConfig>::CallbackType cb = boost::bind(&VMCLNode::reconfigureCallback, this, _1, _2);
		dsrv_->setCallback(cb);

	}

	VMCLNode::~VMCLNode()
	{
	}

	void VMCLNode::reconfigureCallback(const vmcl::VMCLConfig& param, uint32_t level)
	{
		correct_distance_ = param.correct_distance;
		move_mean_window_num_ = param.move_mean_window_num;
		low_pass_coefficient_ = param.low_pass_coefficient;
		pose_filter_->setWindowNum(move_mean_window_num_);
		// pose_filter_->setFilterCoefficient(low_pass_coefficient_);
		debug_eular_.x = param.debug_roll;
		debug_eular_.y = param.debug_ptich;
		debug_eular_.z = param.debug_yaw;
	}

	potbot_lib::Pose toPose(const geometry_msgs::Pose& p) {
		potbot_lib::Pose pose;
		pose.position.x = p.position.x;
		pose.position.y = p.position.y;
		pose.position.z = p.position.z;

		double roll,pitch,yaw;
		potbot_lib::utility::get_rpy(p.orientation, roll,pitch,yaw);
		pose.rotation.x = roll;
		pose.rotation.y = pitch;
		pose.rotation.z = yaw;
		
		return pose;
	}

	Marker VMCLNode::getMarkerTruth(int id)
	{
		geometry_msgs::PoseStamped marker_msg = potbot_lib::utility::get_frame_pose(*tf_buffer_, source_frame_, "marker_" + std::to_string(id));

		Marker marker;
		marker.id = id;
		marker.frame_id = source_frame_;
		marker.pose = toPose(marker_msg.pose);

		return marker;
	}

	Eigen::Affine3d poseMsgToAffine(const geometry_msgs::Pose& pose) 
	{
		Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
		Eigen::Quaterniond quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
		Eigen::Matrix3d rotation = quaternion.toRotationMatrix();

		Eigen::Affine3d affineTransform = Eigen::Affine3d::Identity();
		affineTransform.translation() = translation;
		affineTransform.linear() = rotation;

		return affineTransform;
	}

	void VMCLNode::imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg)
	{	
		std::string frame_id_image = rgb_msg->header.frame_id;
		std::vector<Marker> observed_markers;
		getMarkerCoords(rgb_msg, depth_msg, info_msg, observed_markers);
		publishMarker(observed_markers);
		// ROS_INFO_STREAM(observed_markers.size());

		// if (!observed_markers.empty()) potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(observed_markers.front().pose));

		std::vector<int> observed_marker_ids;
		observed_marker_ids_pre_.resize(1);
		// if (tf_buffer_->canTransform("robot_0/odom","map",ros::Time::now()))
		// 	pose_difference_ = potbot_lib::utility::get_frame_pose(*tf_buffer_,"map","robot_0/odom").pose;
		for (const auto& marker:observed_markers)
		{
			// double margin = 0.1;
			// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker.pose));
			if (marker.pose.position.norm() <= correct_distance_)// &&
				// marker.pose.rotation.x > -margin && marker.pose.rotation.x < margin && 
				// marker.pose.rotation.y > -margin && marker.pose.rotation.y < margin && 
				// marker.pose.rotation.z > -M_PI-margin && marker.pose.rotation.z < -M_PI+margin)
			{

				observed_marker_ids.push_back(marker.id);
				if (!potbot_lib::utility::is_containing(marker.id, observed_marker_ids_pre_) || true)
				{
					// 世界座標系マーカー（既知）
					Eigen::Affine3d marker_world = getMarkerTruth(marker.id).pose.to_affine();
					
					geometry_msgs::PoseStamped marker_sensor_msg = potbot_lib::utility::get_tf(*tf_buffer_, marker.to_msg(), frame_id_camera_);
					// センサー座標系マーカー（既知）
					Eigen::Affine3d marker_sensor = poseMsgToAffine(marker_sensor_msg.pose);
					// marker_sensor = marker.pose.to_affine();
					// marker_sensor =  potbot_lib::Pose{1.5,0.5,0.2,0,0,M_PI}.to_affine();
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker_sensor));

					// 世界座標系センサー（未知）
					Eigen::Affine3d sensor_world = marker_world * marker_sensor.inverse();
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(sensor_world));
					pose_filter_->setData(potbot_lib::Pose(sensor_world));
					sensor_world = pose_filter_->mean().to_affine();
					// sensor_world = pose_filter_->filter().to_affine();

					nav_msgs::Odometry estimate_odometry;
					estimate_odometry.header.frame_id = source_frame_;
					estimate_odometry.child_frame_id = frame_id_camera_;
					estimate_odometry.header.stamp = ros::Time::now();
					estimate_odometry.pose.pose = potbot_lib::utility::get_pose(sensor_world);
					pub_estimate_odometry_.publish(estimate_odometry);

					geometry_msgs::PoseStamped camera_pose_msg = potbot_lib::utility::get_frame_pose(*tf_buffer_, frame_id_odom_, frame_id_camera_);

					//オドメトリー座標系センサー
					Eigen::Affine3d sensor_odom = toPose(camera_pose_msg.pose).to_affine();
					
					// Sw=So*Ow
					// Ow=Sw*So^-1
					Eigen::Affine3d odom_world = sensor_world*sensor_odom.inverse();
					pose_difference_ = potbot_lib::utility::get_pose(odom_world);
					
					pose_difference_ = potbot_lib::utility::get_pose(
						pose_difference_.position.x,
						pose_difference_.position.y,
						0,0,0,
						tf2::getYaw(pose_difference_.orientation)
					);

					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker.pose));

				}
				observed_marker_ids_pre_[0] = marker.id;

			}

		}
		// observed_marker_ids_pre_ = observed_marker_ids;

	}

	void VMCLNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
	{
		encoder_odometry_ = *msg;
		frame_id_odom_ = encoder_odometry_.header.frame_id;
		encoder_odometry_.header.stamp = ros::Time::now();

		potbot_lib::utility::broadcast_frame(dynamic_br_, source_frame_, encoder_odometry_.header.frame_id, pose_difference_);
		// potbot_lib::utility::broadcast_frame(dynamic_br_, encoder_odometry_.header.frame_id, encoder_odometry_.child_frame_id, encoder_odometry_.pose.pose);

		// pub_odometry_.publish(encoder_odometry_);

		// updateParticles();
		// resampling();
		// publishParticles();
	}

	void VMCLNode::iniposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
	{
		if (msg->header.frame_id == source_frame_)
		{
			// Rw=RoOw
			// Ow=RwRo^-1
			Eigen::Affine3d target_world = toPose(msg->pose.pose).to_affine();
			Eigen::Affine3d target_odom = toPose(encoder_odometry_.pose.pose).to_affine();
			Eigen::Affine3d odom_world = target_world*target_odom.inverse();
			pose_difference_ = potbot_lib::utility::get_pose(odom_world);
		}
	}

	bool VMCLNode::getMarkerCoords(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg, std::vector<Marker>& markers)
	{
		cv_bridge::CvImagePtr bridgeImage;//クラス::型//cv_brigeは画像変換するとこ
		cv_bridge::CvImagePtr bridgedepthImage;//クラス::型//cv_brigeは画像変換するとこ
		cv::Mat RGBimage,depthimage,image;

		std::string frame_id_image = rgb_msg->header.frame_id;
		
		try
		{//MAT形式変換
			bridgeImage=cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);//MAT形式に変える
			//ROS_INFO("callBack");//printと秒数表示
		}
		catch(cv_bridge::Exception& e) //エラー処理
		{//エラー処理(失敗)成功ならスキップ
			std::cout<<"depth_image_callback Error \n";
			ROS_ERROR("Could not convert from '%s' to 'BGR8'.",rgb_msg->encoding.c_str());
			return false;
		}

		try
		{//MAT形式変換
			bridgedepthImage=cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);//MAT形式に変える
			//ROS_INFO("callBack");
		}//printと秒数表示
		catch(cv_bridge::Exception& e) //エラー処理
		{//エラー処理(失敗)成功ならスキップ
			std::cout<<"depth_image_callback Error \n";
			ROS_ERROR("Could not convert from '%s' to '32FC1'.",depth_msg->encoding.c_str());
			return false;
		}

		cv::Mat img_src = bridgeImage->image.clone();//image変数に変換した画像データを代入  生の画像
		cv::Mat img_depth = bridgedepthImage->image.clone();//image変数に変換した画像データを代入

		markers.clear();
		cv::Mat img_dst;	//arucoマーカー検出
		img_src.copyTo(img_dst);

		cv::Mat distCoeffs = cv::Mat(info_msg->D.size(), 1, CV_64F);
		for (size_t i = 0; i < info_msg->D.size(); ++i) {
			distCoeffs.at<double>(i, 0) = info_msg->D[i];
		}

		// カメラ行列 K (3x3行列)
		cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, (void*)info_msg->K.data()).clone();

		// 回転行列 R (3x3行列)
		cv::Mat R = cv::Mat(3, 3, CV_64F, (void*)info_msg->R.data()).clone();

		// 投影行列 P (3x4行列)
		cv::Mat projectionMatrix = cv::Mat(3, 4, CV_64F, (void*)info_msg->P.data()).clone();

		//マーカ辞書作成 6x6マスのマーカを250種類生成
		cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

		//charucoボード生成 10x7マスのチェスボード、グリッドのサイズ0.04f、グリッド内マーカのサイズ0.02f
		cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(10, 7, 0.04f, 0.02f, dictionary);

		//マーカー検出時メソッドを指定
		cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f> > markerCorners;
		cv::aruco::detectMarkers(img_src, board->dictionary, markerCorners, markerIds, params);

		std::vector<cv::Vec3d> rvecs,tvecs;//マーカーの姿勢(回転ベクトル、並進ベクトル)

		double cx = cameraMatrix.at<double>(0,2);
		double cy = cameraMatrix.at<double>(1,2);
		double fx = cameraMatrix.at<double>(0,0);
		double fy = cameraMatrix.at<double>(1,1);
		double tx = projectionMatrix.at<double>(0,3);
		double ty = projectionMatrix.at<double>(1,3);
		double tz = projectionMatrix.at<double>(2,3);

		//マーカー観測可能
		if (markerIds.size() > 0) 
		{
			cv::aruco::drawDetectedMarkers(img_dst, markerCorners, markerIds);//マーカー位置を描画
			//cv::aruco::drawDetectedMarkers(img_tate, markerCorners, markerIds);//マーカー位置を描画
			cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);//マーカーの姿勢推定

			for(int i=0;i<markerIds.size();i++)
			{
				int id = markerIds.at(i);
				const auto& corners = markerCorners[i];
				float mcx = (corners[0].x+corners[1].x)/2;//マーカー中心座標(x座標)
				float mcy = (corners[0].y+corners[2].y)/2;//マーカー中心座標(y座標)
				cv::circle(img_dst, cv::Point(mcx,mcy), 3, Scalar(0,255,0),  -1, cv::LINE_AA);//緑点
				cv::aruco::drawAxis(img_dst, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);//マーカーの姿勢描写
				// ROS_INFO_STREAM("r"<<rvecs[i]/M_PI*180);
				// ROS_INFO_STREAM("t"<<tvecs[i]);

				//画像→カメラ座標変換(マーカーの中心座標を使用)
				// float depth = img_depth.at<float>(cv::Point(mcx,mcy))+100;
				float depth = img_depth.at<float>(cv::Point(mcx,mcy));

				//Depthが取得できないコーナーを削除する+Depthの外れ値を除く
				if(depth>0&&depth<10000)
				{
					double x = (mcx - cx) / fx;//ここで正規化座標もしてる
					double y = (mcy - cy) / fy;

					//camera_info.K[0]=615.337,camera_info.K[2]=324.473,camera_info.K[4]=615.458,camera_info.K[5]=241.696//内部パラメータ
					//324.473:画像横サイズ（半分）、241.696:画像サイズ（半分）、615.337:焦点距離、615.458:焦点距離

					Marker m;
					m.id = id;
					m.frame_id = frame_id_image;
					m.pose.position.x = depth*x/depth_scaling_;
					m.pose.position.y = depth*y/depth_scaling_;
					m.pose.position.z = depth/depth_scaling_;

					// m.pose.rotation.x = rvecs[i][0]+debug_eular_.x;
					// m.pose.rotation.y = rvecs[i][1]+debug_eular_.y;
					// m.pose.rotation.z = rvecs[i][2]+debug_eular_.z;
					m.pose.rotation.x = rvecs[i][0]-M_PI_2;
					m.pose.rotation.y = rvecs[i][1]+M_PI_2;
					m.pose.rotation.z = rvecs[i][2];

					markers.push_back(m);

					// ROS_INFO("%d, %f, %f, %f", id, m.pose.position.x, m.pose.position.y, m.pose.position.z);

				}
			}
		}

		cv::imshow("aruco marker", img_dst);
		cv::waitKey(1);
		return true;
	}

	void VMCLNode::publishMarker(const std::vector<Marker>& markers)
	{
		visualization_msgs::MarkerArray markerarray_msg;
		for (const auto& m:markers)
		{
			visualization_msgs::Marker marker_msg;
			marker_msg.ns = "aruco";
			marker_msg.header.frame_id = m.frame_id;
			marker_msg.header.stamp = ros::Time::now();
			marker_msg.id = m.id;
			marker_msg.lifetime = ros::Duration(0.2);
			marker_msg.pose = potbot_lib::utility::get_pose(m.pose);
			// potbot_lib::utility::print_pose(marker_msg.pose);
			marker_msg.type = visualization_msgs::Marker::CUBE;
			marker_msg.scale.x = 0.025;
			marker_msg.scale.y = 0.25;
			marker_msg.scale.z = 0.25;
			marker_msg.color = potbot_lib::color::get_msg(marker_msg.id);

			visualization_msgs::Marker marker_axes = marker_msg;
			marker_axes.ns = "axes";
			marker_axes.type = visualization_msgs::Marker::LINE_LIST;
			marker_axes.scale.x = 0.01;
			marker_axes.scale.y = 0.001;
			marker_axes.scale.z = 0.001;

			geometry_msgs::Point orig = potbot_lib::utility::get_point(0,0,0);
			geometry_msgs::Point unitx = potbot_lib::utility::get_point(0.3,0,0);
			geometry_msgs::Point unity = potbot_lib::utility::get_point(0,0.3,0);
			geometry_msgs::Point unitz = potbot_lib::utility::get_point(0,0,0.3);

			std_msgs::ColorRGBA red = potbot_lib::color::get_msg("red");
			std_msgs::ColorRGBA green = potbot_lib::color::get_msg("green");
			std_msgs::ColorRGBA blue = potbot_lib::color::get_msg("blue");

			marker_axes.points.push_back(orig);
			marker_axes.points.push_back(unitx);
			marker_axes.colors.push_back(red);
			marker_axes.colors.push_back(red);

			marker_axes.points.push_back(orig);
			marker_axes.points.push_back(unity);
			marker_axes.colors.push_back(green);
			marker_axes.colors.push_back(green);

			marker_axes.points.push_back(orig);
			marker_axes.points.push_back(unitz);
			marker_axes.colors.push_back(blue);
			marker_axes.colors.push_back(blue);

			markerarray_msg.markers.push_back(marker_msg);
			markerarray_msg.markers.push_back(marker_axes);
		}

		pub_observed_marker_.publish(markerarray_msg);
	}

	// void VMCLNode::publishParticles()
	// {
		// geometry_msgs::PoseArray particles_msg;
		// particles_msg.header.stamp = ros::Time::now();//追加
		// particles_msg.header.frame_id = source_frame_;//追加
		// for(const auto& p:particles_) particles_msg.poses.push_back(potbot_lib::utility::get_pose(p));
		// pub_particles_.publish(particles_msg);
	// }
}

int main(int argc,char **argv)
{
	ros::init(argc,argv,"vmcl_node");
	
	tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
	vmcl::VMCLNode vmcl(&tfBuffer);

  	ros::spin();
			
	return 0;
}