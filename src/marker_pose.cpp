#include <vmcl/vmcl.h>

ros::Publisher g_pub_observed_marker, g_pub_observed_marker_img;
double g_depth_scaling = 1000;

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

bool getMarkerCoords(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg, std::vector<vmcl::Marker>& markers)
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
			cv::circle(img_dst, cv::Point(mcx,mcy), 3, cv::Scalar(0,255,0),  -1, cv::LINE_AA);//緑点
			cv::aruco::drawAxis(img_dst, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);//マーカーの姿勢描写
			// ROS_INFO_STREAM("r"<<rvecs[i]/M_PI*180);
			// ROS_INFO_STREAM("r"<<rvecs[i]);
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

				vmcl::Marker m;
				m.id = id;
				m.frame_id = frame_id_image;

				// m.pose.position.x = tvecs[i][0];
				// m.pose.position.y = tvecs[i][1];
				// m.pose.position.z = tvecs[i][2];

				m.pose.position.x = depth*x/g_depth_scaling;
				m.pose.position.y = depth*y/g_depth_scaling;
				m.pose.position.z = depth/g_depth_scaling;

				// m.pose.rotation.x = rvecs[i][0]+debug_eular_.x;
				// m.pose.rotation.y = rvecs[i][1]+debug_eular_.y;
				// m.pose.rotation.z = rvecs[i][2]+debug_eular_.z;

				// m.pose.rotation.x = rvecs[i][0]-M_PI_2;
				// m.pose.rotation.y = rvecs[i][1]+M_PI_2;
				// m.pose.rotation.z = rvecs[i][2];

				// m.pose.rotation.x = rvecs[i][0];
				// m.pose.rotation.y = rvecs[i][1];
				// m.pose.rotation.z = rvecs[i][2];

				cv::Mat rotation_matrix;
				cv::Rodrigues(rvecs[i], rotation_matrix);
				tf2::Matrix3x3 tf_rotation(
					rotation_matrix.at<double>(0,0), rotation_matrix.at<double>(0,1), rotation_matrix.at<double>(0,2),
					rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(1,1), rotation_matrix.at<double>(1,2),
					rotation_matrix.at<double>(2,0), rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2)
				);
				tf2::Quaternion tf_quat;
				tf_rotation.getRotation(tf_quat);
				geometry_msgs::Quaternion orientation;
				orientation.x = tf_quat.x();
				orientation.y = tf_quat.y();
				orientation.z = tf_quat.z();
				orientation.w = tf_quat.w();
				potbot_lib::utility::get_rpy(orientation, m.pose.rotation.x, m.pose.rotation.y, m.pose.rotation.z);

				markers.push_back(m);

				// ROS_INFO("%d, %f, %f, %f", id, m.pose.position.x, m.pose.position.y, m.pose.position.z);

			}
		}
	}

	std_msgs::Header header;
	header.stamp = ros::Time::now();
	cv_bridge::CvImage cv_img_dst(header, "bgr8", img_dst);
	g_pub_observed_marker_img.publish(cv_img_dst.toImageMsg());
	// cv::imshow("aruco marker", img_dst);
	// cv::waitKey(1);
	return true;
}

void publishMarker(const std::vector<vmcl::Marker>& markers)
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
		marker_msg.scale.x = 0.25;
		marker_msg.scale.y = 0.25;
		marker_msg.scale.z = 0.025;
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

	g_pub_observed_marker.publish(markerarray_msg);
}

void imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg)
{	
	std::vector<vmcl::Marker> observed_markers_camera;
	getMarkerCoords(rgb_msg, depth_msg, info_msg, observed_markers_camera);
	publishMarker(observed_markers_camera);
}

int main(int argc,char **argv)
{
	ros::init(argc,argv,"marker_pose");
	
	ros::NodeHandle pnh("~");

	pnh.getParam("depth_scaling", g_depth_scaling);

	message_filters::Subscriber<sensor_msgs::Image> sub_rgb;
	message_filters::Subscriber<sensor_msgs::Image> sub_depth;
	message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info;
	ros::NodeHandle nh_sub;
	sub_rgb.subscribe(nh_sub, "color/image_raw", 1);
	sub_depth.subscribe(nh_sub, "depth/image_raw", 1);
	sub_info.subscribe(nh_sub, "color/camera_info", 1);

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
	boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;

	sync.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), sub_rgb, sub_depth, sub_info));
	sync->registerCallback(boost::bind(&imageCallback, _1, _2, _3));

	ros::NodeHandle nhPub;
	g_pub_observed_marker = nhPub.advertise<visualization_msgs::MarkerArray>("debug/observed_marker",1);
	g_pub_observed_marker_img = nhPub.advertise<sensor_msgs::Image>("debug/observed_marker/image", 1);

  	ros::spin();
			
	return 0;
}