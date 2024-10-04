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

		noise_params_ = {
			{m_vv,v_vv},	//直進で生じる道のり
			{m_vw,v_vw},	//回転で生じる道のり
			{m_wv,v_wv},
			{m_ww,v_ww}
		};
		
		//Realsensesの時(roslaunch realsense2_camera rs_camera.launch align_depth:=true)(Depth修正版なのでこっちを使うこと)
		sub_odom_ = nh_sub_.subscribe("odom", 1, &VMCLNode::odomCallback, this);
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

		initParticles();

		pose_filter_ = new potbot_lib::filter::MoveMeanPose(move_mean_window_num_);
		
		dsrv_ = new dynamic_reconfigure::Server<vmcl::VMCLConfig>(pnh);
		dynamic_reconfigure::Server<vmcl::VMCLConfig>::CallbackType cb = boost::bind(&VMCLNode::reconfigureCallback, this, _1, _2);
		dsrv_->setCallback(cb);

		pose_diffetence_ = potbot_lib::utility::get_pose();
	}

	VMCLNode::~VMCLNode()
	{
	}

	void VMCLNode::reconfigureCallback(const vmcl::VMCLConfig& param, uint32_t level)
	{
		correct_distance_ = param.correct_distance;
		move_mean_window_num_ = param.move_mean_window_num;
		pose_filter_->setWindowNum(move_mean_window_num_);
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
		// 	pose_diffetence_ = potbot_lib::utility::get_frame_pose(*tf_buffer_,"map","robot_0/odom").pose;
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
					// marker_sensor =  potbot_lib::Pose{1.5,0.5,0.2,0,0,M_PI}.to_affine();
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker_sensor));

					// 世界座標系センサー（未知）
					Eigen::Affine3d sensor_world = marker_world * marker_sensor.inverse();
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(sensor_world));
					pose_filter_->setData(potbot_lib::Pose(sensor_world));
					sensor_world = pose_filter_->mean().to_affine();

					nav_msgs::Odometry estimate_odometry;
					estimate_odometry.header.frame_id = source_frame_;
					estimate_odometry.child_frame_id = frame_id_camera_;
					estimate_odometry.header.stamp = ros::Time::now();
					estimate_odometry.pose.pose = potbot_lib::utility::get_pose(sensor_world);
					pub_estimate_odometry_.publish(estimate_odometry);

					geometry_msgs::PoseStamped camera_pose_msg = potbot_lib::utility::get_frame_pose(*tf_buffer_, source_frame_, frame_id_camera_);
					Eigen::Affine3d camera_pose = toPose(camera_pose_msg.pose).to_affine();
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(camera_pose));

					Eigen::Vector3d translation_diff = sensor_world.translation() - camera_pose.translation();
					// ROS_INFO_STREAM(translation_diff.transpose());

					pose_diffetence_.position.x = translation_diff.x();
					pose_diffetence_.position.y = translation_diff.y();
					// pose_diffetence_.position.z = translation_diff.z();
					pose_diffetence_.position.z = 0;
					// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker.pose));

				}
				observed_marker_ids_pre_[0] = marker.id;

			}

		}
		// observed_marker_ids_pre_ = observed_marker_ids;
		

		// for (const auto& p:particles_) ROS_INFO("%f, %f, %f", p.position.x, p.position.y, p.rotation.z);
		// ROS_INFO("%f, %f, %f", particles_[0].position.x, particles_[0].position.y, particles_[0].rotation.z);
		updateParticles();

		double xsum=0,ysum=0,thsum=0;
		for (const auto& p:particles_)
		{
			xsum+=p.position.x;
			ysum+=p.position.y;
			thsum+=p.rotation.z;
		}
		
		double xhat = xsum/particles_.size();
		double yhat = ysum/particles_.size();
		double thhat = thsum/particles_.size();
		
		// nav_msgs::Odometry estimate_odometry;
		// estimate_odometry.header.frame_id = source_frame_;
		// estimate_odometry.header.stamp = ros::Time::now();
		// estimate_odometry.pose.pose = potbot_lib::utility::get_pose(xhat, yhat, 0, 0, 0, thhat);
		// pub_estimate_odometry_.publish(estimate_odometry);
		
		std::vector<double> ss;//重みのリスト

		//(最大尤度と合計尤度)また最大尤度を取るパーティクル番号とその座標を表示させるプログラム(山口追記)※一応
		double totalLikelihood = 0.0; //パーティクル群の尤度の総和
		double maxLikelihood = 0.0;
		int maxLikelihoodParticleIdx = 0; //最大尤度を持つパーティクル識別番号

		for(int i = 0; i < particle_weight_.size(); i++)
		{
			if(i == 0)
			{
				maxLikelihood = particle_weight_[i];
				maxLikelihoodParticleIdx = 0;
			}
			else if (maxLikelihood < particle_weight_[i])
			{
				maxLikelihood = particle_weight_[i];
				maxLikelihoodParticleIdx = i;
			}
			totalLikelihood += particle_weight_[i];
		}
		//追加システム(最大尤度と合計尤度、重みの正規化と自己位置推定)終了

		//リサンプリング
		std::vector<double> sd;//重みの累積和
		double sum=0;

		for (int i = 0; i < particle_num_; i++)
		{
			if (particle_weight_[i] < 1e-100)
			{
				particle_weight_[i] = particle_weight_[i] + 1e-100;
			}
			
			ss.push_back(particle_weight_[i]);
		}
		
		for (int i = 0; i <particle_num_ ; i++)
		{
			sum+=ss[i];
			sd.push_back(sum);
			//std::cout << "sd=" <<sd[i]<< std::endl;
		}//累積和　sd={1,3,6,10,15}
		//std::cout << "sum=" <<sum<< std::endl;

		double step=sd[particle_num_-1]/particle_num_;//(重みの合計)/(パーティクルの合計)

		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<double> distr(0,step);
		double r=distr(eng);//0~stepの間でランダムな値を抽出する関数(おそらく小数付き)
		//std::cout << "r=" <<r<< std::endl;

		int cur_pos=0;
		int math=0;
		std::vector<potbot_lib::Pose> ps;//新たに抽出するパーティクルのリスト

		while (ps.size() < particle_num_)//もとのパーティクル数と一致するまで
		{
			if (r < sd[cur_pos])//重みが幅より大きい場合
			{
				ps.push_back(particles_[cur_pos]);
				r+=step;
			}
			else
			{
				cur_pos+=1;
			}
		}
		//作り上げたpsの配列をもとのパーティクルの配列にコピーする処理
		copy(ps.begin(),ps.end(),particles_.begin());
		
		for (int i = 0; i < particle_num_; i++)
		{
			particle_weight_.at(i)=1/particle_num_;
		}
		//リサンプリング（終わり）

		geometry_msgs::PoseArray particles_msg;
		particles_msg.header.stamp = ros::Time::now();//追加
		particles_msg.header.frame_id = source_frame_;//追加
		for(const auto& p:particles_) particles_msg.poses.push_back(potbot_lib::utility::get_pose(p));
		pub_particles_.publish(particles_msg);

	}

	void VMCLNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
	{
		encoder_odometry_ = *msg;
		// encoder_odometry_.header.stamp = ros::Time::now();
		static tf2_ros::TransformBroadcaster dynamic_br;
		potbot_lib::utility::broadcast_frame(dynamic_br, source_frame_, encoder_odometry_.header.frame_id, pose_diffetence_);
		// potbot_lib::utility::broadcast_frame(dynamic_br, encoder_odometry_.header.frame_id, encoder_odometry_.child_frame_id, encoder_odometry_.pose.pose);

		// pub_odometry_.publish(encoder_odometry_);
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

	void VMCLNode::initParticles()
	{
		particles_.resize(particle_num_);
		particle_weight_.resize(particle_num_,1/particle_num_);
		breez_.resize(particle_num_);
		greed_.resize(particle_num_);

		noise_.resize(noise_params_.size());
		std::random_device seed;
		std::mt19937 engine(seed());
		std::vector<std::normal_distribution<>> dists(noise_params_.size());
		for (size_t i = 0; i < noise_params_.size(); i++)
		{
			std::normal_distribution<> dist(noise_params_[i][0],noise_params_[i][1]);
			dists[i] = dist;
		}
		
		for (size_t i = 0; i < noise_.size(); i++)
		{
			for (size_t j = 0; j < particle_num_; j++)
			{
				noise_[i].push_back(dists[i](engine));
			}
		}
	}

	void VMCLNode::updateParticles()
	{	
		ros::Time timestamp_now = ros::Time::now();
		static ros::Time timestamp_pre = timestamp_now;
		double dt = timestamp_now.toSec() - timestamp_pre.toSec();
		timestamp_pre = timestamp_now;

		double v = encoder_odometry_.twist.twist.linear.x;
		double omega = encoder_odometry_.twist.twist.angular.z;

		if(dt>0)
		{
			for (int i = 0; i < particle_num_; i++)
			{
				//目標値(Des)にノイズを入れることで擬似的に実効値(Act)を作り出している
				// breez_[i]=velocity_command_.linear.x+noise_[0][i]*sqrt(abs(velocity_command_.linear.x)/realsec)+noise_[1][i]*sqrt(abs(velocity_command_.angular.z)/realsec);//ノイズ付き速度(山口先輩)
				// greed_[i]=velocity_command_.angular.z+noise_[2][i]*sqrt(abs(velocity_command_.linear.x)/realsec)+noise_[3][i]*sqrt(abs(velocity_command_.angular.z)/realsec);//ノイズ付き角速度（山口先輩）
				
				breez_[i]=v+noise_[0][i]*sqrt(abs(v)/dt)+noise_[1][i]*sqrt(abs(omega)/dt);//ノイズ付き速度(エンコーダ基準)（鈴木先輩）
				greed_[i]=omega+noise_[2][i]*sqrt(abs(v)/dt)+noise_[3][i]*sqrt(abs(omega)/dt);//ノイズ付き角速度（鈴木先輩）

				// ROS_INFO("%d, %f, %f", i, breez_[i], greed_[i]);
			}
		}

		// double breez_min = *min_element(begin(breez_), end(breez_));
		// double breez_max = *max_element(begin(breez_), end(breez_));
		// double greed_min = *min_element(begin(greed_), end(greed_));
		// double greed_max = *max_element(begin(greed_), end(greed_));

		// ROS_INFO("linear, min:%f, max:%f", breez_min, breez_max);
		// ROS_INFO("angular, min:%f, max:%f", greed_min, greed_max);

		// ROS_INFO("noise");
		// for (int i = 0; i < noise_.size(); i++) 
		// {
		// 	double noise_min = *min_element(begin(noise_[i]), end(noise_[i]));
		// 	double noise_max = *max_element(begin(noise_[i]), end(noise_[i]));
		// 	ROS_INFO("	%d, min:%f, max:%f", i, noise_min, noise_max);
		// }

		for(int v=0;v<particle_num_;v++)
		{
			double th = particles_[v].rotation.z;
			if(greed_[v]<1e-10)
			{
				particles_[v].position.x += dt*breez_[v]*cos(th);
				particles_[v].position.y += dt*breez_[v]*sin(th);
				particles_[v].rotation.z += greed_[v]*dt;
			}
			else
			{
				particles_[v].position.x += (breez_[v]/greed_[v])*(sin(th+greed_[v]*dt)-sin(th));
				particles_[v].position.y += (breez_[v]/greed_[v])*(-cos(th+greed_[v]*dt)+cos(th));
				particles_[v].rotation.z += greed_[v]*dt;
			}
		}
	}
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