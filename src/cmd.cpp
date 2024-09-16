//特徴点検出のテストプログラム
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
#include <opencv2/ximgproc/fast_line_detector.hpp>//FLD
#include <time.h> //処理の時間を出力する
#include <sys/time.h>
#include <random>
#include <geometry_msgs/PoseStamped.h> //tf
#include <tf/transform_broadcaster.h>  //tf
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <opencv2/aruco/charuco.hpp> //マーカー検出
#include <nav_msgs/Path.h>           //経路情報を記録する
#include <array>

#include <geometry_msgs/Twist.h> //ロボットの指令値(速度)用ヘッダー
#include <nav_msgs/Odometry.h>
// #include <fstream>
#include <std_msgs/Float32MultiArray.h>
#include<sensor_msgs/Joy.h>
#include <geometry_msgs/PoseArray.h>

#include <potbot_lib/utility_ros.h>

using namespace std;
using namespace cv;

struct Marker
{
	int id = 0;
	potbot_lib::Pose pose;
};

class VMCLNode
{
	private:
		std::string win_src = "src";
		std::string win_dst = "dst";
		std::string win_dst2 = "dst2";
		std::string win_dstfld = "dstfld";
		std::string win_fld = "fld";
		std::string win_point = "point";
		std::string win_fld_ty="FLD_TY";
		std::string win_line2 = "line2";
		std::string win_line3 = "line3";
		std::string win_line4 = "line4";

		std::string source_frame = "map"; //mapフレーム

		std::string win_graph = "graph";
		std::string win_tate = "tate";

		ros::Subscriber odom_sub;    //ロボットオドメトリー受信用
		ros::Publisher pub;          //速度送信用

		ros::Publisher estimate_odometry_pub;
		ros::Publisher particles_pub;//パーティクル
		ros::Publisher pub_odometry_;

		bool time0 = false; //カルマン推定データ初回回避用(カルマン動作した時TRUE)
		bool yoko_histgram = false;//ヨコ線のヒストグラム(true実行)
		bool tate_histgram = false;//タテ線のヒストグラム(true実行)
		bool reset = true;
		bool Tracking=false;
		bool X_25 = false;
		bool TH_90 = false;
		bool Y_05 = false;
		bool Y_051 = false;
		bool Y_052 = false;
		bool InitialNoise = false;

		cv::Mat img_tate;//縦線関連の画像表示
		vector<cv::Point2f> points_prev, points_curr;       //特徴点定義
		vector<cv::Point3f> camera_point_p, camera_point_c; //特徴点定義

		sensor_msgs::CameraInfo camera_info;                //CameraInfo受け取り用
		geometry_msgs::Twist robot_velocity;                //指令速度

		int kaisu = 0, kaisuM1 = 0, kaisuV1 = 0;

		ros::WallTime wall_begin = ros::WallTime::now(); //プログラム開始時の時間
		ros::WallDuration wall_prev;                     //一つ前の時間
		ros::WallDuration wall_systemtime;               //サンプリング時間
		struct timeval startTime, endTime;     // 構造体宣言
		struct timeval startTime2, endTime2;   // 構造体宣言
		struct timeval startTimeV1, endTimeV1; // 構造体宣言
		struct timeval startTimeM1, endTimeM1; // 構造体宣言

		float realsec;                 //サンプリング時間（C++)
		float ALLrealsec;              //サンプリング時間（C++)
		float realsecV1, ALLrealsecV1; //サンプリング時間（C++)
		float realsecM1, ALLrealsecM1; //サンプリング時間（C++)

		int template_size=10;
		double VX,omegaZ,LX,THZ,LY;//ロボットの指令値パラメータ
		double cropx=10*2.5;//予測範囲の範囲(template_size*n)
		double cropy=10*2.5;//予測範囲の範囲(template_size*n)

		cv::Mat frame,image_curr, image_prev,img_dst,
		img_dst2, //構造線付き画像
		img_dstfld,//マーカーと構造線付き画像
		img_1;
		cv::Mat img_FLD_TY,img_FLD_T,img_FLD_Y,img_graph;
		double Average_tate_theta,Average_tate_theta_Y;//最大クラスタ内の平均角度を求める

		double pnum=100;//パーティクル個数
		double balance=10000;//尤度調整用
		//std_msgs::Float32MultiArray asd;//多変量正規分布
		double roll, pitch, yaw;                                //クオータニオン→オイラー角変換用

		double Act_RobotX =0, Act_RobotY = 0, Act_RobotTH = 0; //ロボットの状態方程式(実際の状態)
		double Des_RobotX = 0, Des_RobotY = 0, Des_RobotTH = 0; //ロボットの状態方程式(理想状態)
		double aggre_RobotX = 0, aggre_RobotY = 0, aggre_RobotTH = 0; //パーティクルの平均世界座標位置
		double aggrex=0,aggrey=0,aggrez=0;
		double asterx=0,astery=0,asterth=0;
		double standardx=0,standardy=0,standardth=0;

		int maxLikelihoodParticleIdx = 0; //最大尤度を持つパーティクル識別番号
		double totalLikelihood = 0.0; //パーティクル群の尤度の総和
		double averageLikelihood = 0.0; //パーティクル群の尤度の平均
		double effectiveSampleSize = 0.0; //有効サンプル数
		double resampleThreshold = 0.0; //リサンプリングの閾値
		double Likelihood = 0.0; //尤度
		double w = 0.0; //重み
		double odomnoise1 = 2.97694 * std::pow(10, -5); //ロボットの誤差分散値σvv
		double odomnoise2 = 1.26947 * std::pow(10, -4); //ロボットの誤差分散値σvω
		double odomnoise3 = 3.30119 * std::pow(10, -6); //ロボットの誤差分散値σωv 
		double odomnoise4 = 1.11064 * std::pow(10, -3); //ロボットの誤差分散値σωω
		double Particle_Est_RobotX = 0;//重み付き平均によるX座標自己位置
		double Particle_Est_RobotY = 0;//重み付き平均によるY座標自己位置
		double Particle_Est_RobotTH = 0;//重み付き平均による角度

		std::vector<double> NoiseV;
		std::vector<double> NoiseOmega;

		std::vector<potbot_lib::Pose> particles_;	//パーティクルの位置
		std::vector<double> particle_weight_;//各パーティクルに対する重み

		std::vector<double> breez_;//ノイズ付きパーティクル速度
		std::vector<double> greed_;//ノイズ付きパーティクル

		double any[100][4];
		double alpha=0;
		double beta=1e100;
		double Act_RobotV, Des_RobotV;                          //ロボットの速度ベクトル(実測値,指令値)

		int ALLMarker = 40;                                                   //全マーカー個数
		float MC_point_prve[50][4];                                           //一つ前のカメラ座標
		float pixel[50][2], MC_point[50][4], r2, f, ux, uy; //画像→カメラ座標変換
		float depthX_right1,depthX_right2,depthX_right3,depthX_right4,depthX_right5,depth_right_ave;
		float depthX_left1,depthX_left2,depthX_left3,depthX_left4,depthX_left5,depth_left_ave;
		//double CameraLM[50][2];//カメラから見たマーカーまでの距離[0]と角度[1]
		int secadjustment=0;//realsec調整用

		cv::Mat ACT_Robot, DES_Robot; //状態方程式の行列化
		cv::Mat At, Mt, Ft, Cov, Qt, K, Ht, I,hu;

		double LM[1][2], CameraLM[50][2]; //ランドマーク世界座標
		double siguma;                    //センサーの共分散(仮)


		//マーカーの世界座標登録
		cv::Mat_<float> MarkerW[40]; //マーカーの世界座標robot_velocity
		float depth_ideal[40],yoko_ideal[40];


		//縦線のテンプレート取得関連
		vector<cv::Point2f> tate_point_curr,tate_point_prev;//縦線の中点座標(画像座標系)
		double DTPC[100],DTPP[100];//depth_tate_point_curr=DTPC,depth_tate_point_prev=DTPP(Depth取得可能縦線中点の数)
		int DTPC_ok,DTPP_ok;//(Depth取得可能縦線数)
		vector<cv::Point3f> TPCC,TPCP;//tate_point_camera_curr=TPCC,tate_point_camera_prev=TPCP(縦線中点のカメラ座標系)
		cv::Mat TPCC_Templ[100],TPCP_Templ[100];//縦線中点のテンプレート画像

		//テンプレートマッチ用変数
		vector<cv::Point3f> Est_tate_point;//特徴点定義(運動復元
		vector<cv::Point2f> Est_tate_pixel;//特徴点定義(運動復元
		vector<cv::Point2f> MTPC,MTPP;//テンプレートの中心座標(Matching_Tate_Point_Curr=MTPC,Matching_Tate_Point_Prev=MTPP)
		int matchT_curr=0,matchT_prev=0;//マッチングしたテンプレート数
		cv::Mat MTTC[500],MTTP[500];//マッチングしたテンプレート画像キープ用(Matching_Tate_Templ_Curr=MTTC,Matching_Tate_Templ_Prev=MTTP)
		cv::Mat EST_tate_scope[100];//特徴点周囲の切り取った画像(予測範囲画像)
		cv::Mat img_template1,img_master_temp;
		cv::Point min_pt1[100], max_pt1[100];//テンプレートマッチング用変数
		double min_val1[100], max_val1[100];

		cv::Mat MT_curr_Templ[100],MT_prev_Templ[100];//マッチテンプレート画像
		vector<cv::Point2f> MT_curr_pixel,  MT_prev_pixel;//マッチテンプレートの画像座標系
		vector<cv::Point3f> MT_curr_camera, MT_prev_camera;//マッチテンプレートのカメラ座標系
		vector<cv::Point3f> MT_curr_world,  MT_prev_world;//マッチテンプレートの世界座標系
		vector<cv::Point3f> MT_curr_world2, MT_prev_world2;//マッチテンプレートの世界座標系
		double DMT_curr[600],DMT_prev[600];//Depth_Matching_tate_prev(マッチテンプレート座標のDepth)
		int DMT_curr_ok=0,DMT_prev_ok;//マッチテンプレートのDepth取得可能数
		double length;//テンプレートの距離比較用(追加更新動作)
		double DMT_curr_new[600];

		//テンプレートマッチ用変数(3回目動作用)
		cv::Mat MT_curr2_Templ[100];//マッチテンプレート画像
		vector<cv::Point2f> MT_curr2_pixel;//マッチテンプレートの画像座標系
		vector<cv::Point3f> MT_curr2_camera;//マッチテンプレートのカメラ座標系
		vector<cv::Point3f> MT_curr2_world;//マッチテンプレートの世界座標
		double DMT_curr2[600];//Depth_Matching_pixel_prev(マッチテンプレート座標のDepth)
		int DMT_curr2_ok=0;//マッチテンプレートのDepth取得可能数

		int times=0;

		double MTcoix[300],MTcoiy[300];
		vector<cv::Point3f> Est_MT_point;//特徴点定義(運動復元
		vector<cv::Point2f> Est_MT_pixel;//特徴点定義(運動復元
		cv::Mat EST_MT_scope[100];//予測範囲クロップ
		int EST_MT_ok=0;//画面内のテンプレート数
		double CameraLMT[600][2];

		ros::NodeHandle nh_sub_;
		ros::Subscriber sub_odom_;
		message_filters::Subscriber<sensor_msgs::Image> sub_rgb_;
		message_filters::Subscriber<sensor_msgs::Image> sub_depth_;

		// ApproximateTimeポリシーの定義
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
		
		// 同期器の定義
		boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

		nav_msgs::Odometry encoder_odometry_;

		void imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg,const sensor_msgs::Image::ConstPtr& depth_msg);
		void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);

	public:
		VMCLNode(/* args */);
		~VMCLNode();
};

VMCLNode::VMCLNode(/* args */)
{
	//Realsensesの時(roslaunch realsense2_camera rs_camera.launch align_depth:=true)(Depth修正版なのでこっちを使うこと)
	sub_odom_ 		= nh_sub_.subscribe("odom",           1, &VMCLNode::odomCallback, this);

	sub_rgb_.subscribe(nh_sub_, "color/image_raw", 1);//センサーメッセージを使うときは対応したヘッダーが必要
	sub_depth_.subscribe(nh_sub_, "depth/image_raw", 1);

	sync_.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), sub_rgb_, sub_depth_));
	sync_->registerCallback(boost::bind(&VMCLNode::imageCallback, this, _1, _2));

	ros::NodeHandle nhPub;
	estimate_odometry_pub = nhPub.advertise<nav_msgs::Odometry>("estimate_odometry",1000);
	particles_pub = nhPub.advertise<geometry_msgs::PoseArray>("particles_posearray",1000);
	pub_odometry_ = nhPub.advertise<nav_msgs::Odometry>("debug/encoder_odometry",1000);

	//2024-07-31廊下(直進11.0m,速度0.25)直線動作実験
	//rosbag play 2024-07-31-13-10-22.bag//実測値(X:11.400,Y:3.262)
	//rosbag play 2024-07-31-13-34-34.bag//実測値(X:11.050,Y:3.110)
	MarkerW[1]= (cv::Mat_<float>(3, 1) <<3.00, 0.28, 0.902);//実測値(X:,Y:)
	MarkerW[2]= (cv::Mat_<float>(3, 1) <<10.05, 0.28, 0.902);
	MarkerW[3]= (cv::Mat_<float>(3, 1) <<12.30, 0.28, 1.00);
	MarkerW[4]= (cv::Mat_<float>(3, 1) <<10.35, 0.28, 3.00);
	LX=11.0,VX=0.25,omegaZ=1.0,THZ=0.20,LY=3.00;

	// ROS_INFO_STREAM(MarkerW[1]);

	particles_.resize(pnum);
	particle_weight_.resize(pnum,1/pnum);
	breez_.resize(pnum);
	greed_.resize(pnum);
}

VMCLNode::~VMCLNode()
{
}

void VMCLNode::imageCallback(const sensor_msgs::Image::ConstPtr& rgb_msg,const sensor_msgs::Image::ConstPtr& depth_msg)
{	
	//ROS_INFO("callback_functionが呼ばれたよ");
	//サンプリング時間取得(C言語の方法)(こっちのほうが正確らしい)
	gettimeofday(&startTime, NULL);// 開始時刻取得
	if(time0 != false)
	{
		time_t diffsec = difftime(startTime.tv_sec,endTime.tv_sec);    // 秒数の差分を計算
		suseconds_t diffsub = startTime.tv_usec - endTime.tv_usec;      // マイクロ秒部分の差分を計算
		realsec = diffsec+diffsub*1e-6;                          // 実時間を計算
		ALLrealsec=ALLrealsec+realsec;
		//printf("処理の時間=%f\n", realsec);
		//printf("処理時間合計=%f\n", ALLrealsec);
	}

	//サンプリング時間取得(ROS)
	ros::WallTime wall_now = ros::WallTime::now();
	ros::WallDuration wall_duration = wall_now - wall_begin;
	//ROS_INFO("WALL:%u.%09u", wall_duration.sec, wall_duration.nsec);
	wall_systemtime = wall_duration - wall_prev;
	//ROS_INFO("systemtime:%u.%09u", wall_systemtime.sec, wall_systemtime.nsec);
	//std::cout << "wall_prev=" <<wall_prev<< std::endl;//サンプリング時間

	cv_bridge::CvImagePtr bridgeImage;//クラス::型//cv_brigeは画像変換するとこ
	cv_bridge::CvImagePtr bridgedepthImage;//クラス::型//cv_brigeは画像変換するとこ
	cv::Mat RGBimage,depthimage,image;
	
	try
	{//MAT形式変換
		bridgeImage=cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);//MAT形式に変える
		//ROS_INFO("callBack");//printと秒数表示
	}
	catch(cv_bridge::Exception& e) //エラー処理
	{//エラー処理(失敗)成功ならスキップ
		std::cout<<"depth_image_callback Error \n";
		ROS_ERROR("Could not convert from '%s' to 'BGR8'.",rgb_msg->encoding.c_str());
		return ;
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
		return;
	}

	cv::Mat img_src = bridgeImage->image.clone();//image変数に変換した画像データを代入  生の画像
	cv::Mat img_depth = bridgedepthImage->image.clone();//image変数に変換した画像データを代入
	cv::Mat img_gray,img_fld,img_line2,img_line3,img_line4;

	img_src.copyTo(img_dst);
	img_src.copyTo(img_dst2);
	img_src.copyTo(img_dstfld);

	cv::cvtColor(img_src,img_gray,cv::COLOR_RGB2GRAY);

	img_fld = img_src.clone();
	img_fld = cv::Scalar(255,255,255);
	img_FLD_TY = img_src.clone();
	img_FLD_TY = cv::Scalar(255,255,255);
	img_line2 = img_src.clone();
	img_line2 = cv::Scalar(255,255,255);
	img_line3 = img_src.clone();
	img_line3 = cv::Scalar(255,255,255);
	img_line4 = img_src.clone();
	img_line4 = cv::Scalar(255,255,255);
	img_graph = img_src.clone();
	img_graph = cv::Scalar(255,255,255);
	img_tate = img_src.clone();

	//img_dst = image.clone();
	//img_dst = cv::Scalar(255,255,255);
	cv::Mat_<float> intrinsic_K= cv::Mat_<float>(3, 3);

	//マーカー検出+外部パラメータ推定
	//カメラ内部パラメータ読み込み
	cv::Mat cameraMatrix;
	cv::FileStorage fs;
	fs.open("/home/ros/realsense_para.xml", cv::FileStorage::READ);
	fs["intrinsic"]>>cameraMatrix;
	//std::cout << "内部パラメータcameraMatrix=\n" << cameraMatrix << std::endl;
	intrinsic_K=cameraMatrix;

	//カメラの歪みパラメータ読み込み
	cv::Mat distCoeffs;
	cv::FileStorage fd;
	fd.open("/home/ros/realsense_para.xml", cv::FileStorage::READ);
	fd["distortion"]>>distCoeffs;
	//std::cout << "ねじれパラメータdistCoeffs=\n" << distCoeffs << std::endl;

	//マーカ辞書作成 6x6マスのマーカを250種類生成
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

	//charucoボード生成 10x7マスのチェスボード、グリッドのサイズ0.04f、グリッド内マーカのサイズ0.02f
	cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(10, 7, 0.04f, 0.02f, dictionary);

	//マーカー検出時メソッドを指定
	cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

	std::vector<int> markerIds;
	std::vector<std::vector<cv::Point2f> > markerCorners;
	std::vector<Marker> markers;
	cv::aruco::detectMarkers(img_src, board->dictionary, markerCorners, markerIds, params);

	std::vector<cv::Vec3d> rvecs,tvecs;//マーカーの姿勢(回転ベクトル、並進ベクトル)
	cv::Mat_<double> rvecs2[50],jacobian[50];

	float depth0,depthFT;//Depth修正用

	//MCpoint[i][3],MC_point_prve[][][3]はデータ有無の要素（[][][3]=1ならデータ有り,0ならデータ無し)
	if(kaisu==0)//初回のみ全部初期化
	{
		for(int i=0;i<ALLMarker;i++)
		{
			for(int j=0;j<4;j++)
			{
				MC_point[i][3]=0;//全マーカーのデータ確認用要素初期化
				MC_point_prve[i][3]=0;
				//xEst_prev_clP[i]=0;
			}
		}
	}
	//毎回最新pointのみ初期化
	for(int i=0;i<ALLMarker;i++)
	{
		for(int j=0;j<4;j++)
		{
			MC_point[i][3]=0;//全マーカーのデータ確認用要素初期化
		}
	}

	//マーカー観測可能
	if (markerIds.size() > 0) 
	{
		cv::aruco::drawDetectedMarkers(img_dst, markerCorners, markerIds);//マーカー位置を描画
		cv::aruco::drawDetectedMarkers(img_dstfld, markerCorners, markerIds);//マーカー位置を描画
		//cv::aruco::drawDetectedMarkers(img_tate, markerCorners, markerIds);//マーカー位置を描画
		cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);//マーカーの姿勢推定

		for(int i=0;i<markerIds.size();i++)
		{
			//std::cout <<"マーカーの個数:markerIds.size()="<<markerIds.size() << std::endl;//マーカー個数
			//std::cout <<"markerIds("<<i<<")="<< markerIds.at(i) << std::endl;//マーカーID
			// ROS_INFO_STREAM("markerIds("<<i<<")="<< markerIds.at(i));//マーカーID
			int id = markerIds.at(i);
			const auto& corners = markerCorners[i];
			float mcx = (corners[0].x+corners[1].x)/2;//マーカー中心座標(x座標)
			float mcy = (corners[0].y+corners[2].y)/2;//マーカー中心座標(y座標)
			cv::circle(img_dst, cv::Point(mcx,mcy), 3, Scalar(0,255,0),  -1, cv::LINE_AA);//緑点
			cv::circle(img_dstfld, cv::Point(mcx,mcy), 3, Scalar(0,255,0),  -1, cv::LINE_AA);//緑点
			//cv::circle(img_tate, cv::Point(MCx[id],MCy[id]), 3, Scalar(0,255,0),  -1, cv::LINE_AA);//緑点
			cv::aruco::drawAxis(img_dst,cameraMatrix,distCoeffs,rvecs[i], tvecs[i], 0.1);//マーカーの姿勢描写
			cv::aruco::drawAxis(img_dstfld,cameraMatrix,distCoeffs,rvecs[i], tvecs[i], 0.1);
			//cv::aruco::drawAxis(img_tate,cameraMatrix,distCoeffs,rvecs[i], tvecs[i], 0.1);//マーカーの姿勢描写
			cv::Rodrigues(rvecs[i],rvecs2[i],jacobian[i]);//回転ベクトルから回転行列への変換

			//画像→カメラ座標変換(マーカーの中心座標を使用)
			pixel[id][0]=mcx;
			pixel[id][1]=mcy;
			//std::cout <<"MCx["<<id<<"]="<<MCx[id]<<",MCy["<<id<<"]="<<MCy[id]<< std::endl;
			float depth = img_depth.at<float>(cv::Point(pixel[id][0],pixel[id][1]))+100;
			depthX_left1=img_depth.at<float>(cv::Point(50,240))+100;
			depthX_left2=img_depth.at<float>(cv::Point(25,240))+100;
			depthX_left5=img_depth.at<float>(cv::Point(100,240))+100;

			depthX_right1=img_depth.at<float>(cv::Point(590,240))+100;
			depthX_right2=img_depth.at<float>(cv::Point(615,240))+100;
			depthX_right5=img_depth.at<float>(cv::Point(540,240))+100;

			depth_right_ave=(depthX_right1+depthX_right2+depthX_right5)/3;
			depth_left_ave=(depthX_left1+depthX_left2+depthX_left5)/3;
			
			//std::cout<<"depthX"<<depthX<<std::endl;
			cv::circle(img_tate,cv::Point(50,240),3,Scalar(30,30,30),-1,cv::LINE_AA);
			cv::circle(img_tate,cv::Point(25,240),3,Scalar(30,30,30),-1,cv::LINE_AA);
			cv::circle(img_tate,cv::Point(100,240),3,Scalar(30,30,30),-1,cv::LINE_AA);

			cv::circle(img_tate,cv::Point(590,240),3,Scalar(30,30,30),-1,cv::LINE_AA);
			cv::circle(img_tate,cv::Point(615,240),3,Scalar(30,30,30),-1,cv::LINE_AA);
			cv::circle(img_tate,cv::Point(540,240),3,Scalar(30,30,30),-1,cv::LINE_AA);

			//Depthが取得できないコーナーを削除する+Depthの外れ値を除く
			if(depth>0&&depth<10000)
			{
				double x = (pixel[id][0] - 324.473) / 615.337;//ここで正規化座標もしてる
				double y = (pixel[id][1] - 241.696) / 615.458;

				//camera_info.K[0]=615.337,camera_info.K[2]=324.473,camera_info.K[4]=615.458,camera_info.K[5]=241.696//内部パラメータ
				//324.473:画像横サイズ（半分）、241.696:画像サイズ（半分）、615.337:焦点距離、615.458:焦点距離

				//ロボット座標系とカメラ観測座標系ではxの符号が逆なので注意(ロボット座標:Y = カメラ観測座標:-X)
				MC_point[id][0] = -depth * x/1000;//メートル表示変換
				MC_point[id][1] = depth * y/1000;
				MC_point[id][2] = depth/1000;
				MC_point[id][3] = 1;//データ取得可能なら1

				Marker m;
				m.id = id;
				m.pose.position.x = depth/1000;
				m.pose.position.y = -depth * x/1000;
				m.pose.position.z = depth * y/1000;
				markers.push_back(m);

				ROS_INFO("%d, %f, %f, %f", id, m.pose.position.x, m.pose.position.y, m.pose.position.z);

			}

			//マーカーまでの距離と角度を求める(観測値)
			CameraLM[id][0]=sqrt((MC_point[id][0]*MC_point[id][0])+(MC_point[id][2]*MC_point[id][2]));
			CameraLM[id][1]=atan2(MC_point[id][0],MC_point[id][2]);
		
			//std::cout <<"CameraLM["<<id<<"][0]="<<CameraLM[id][0]<< std::endl;//マーカーまでの距離
			//std::cout <<"CameraLM["<<id<<"][1]="<<CameraLM[id][1]*180/M_PI<< std::endl;//マーカーまでの角度
			//camera_robot<< "ALLrealsec=" <<ALLrealsec<< " ,realsec=" <<realsec<<" ,CameraLM["<<id<<"][0]=" <<CameraLM[id][0]<<",CameraLM["<<id<<"][1]=" <<CameraLM[id][1]<<"\n";
		}
	}
	
	//線検出//
	std::vector<cv::Vec4f> lines_fld;//初期定義
	cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector();//特徴線クラスオブジェクトを作成
	fld->detect( img_gray, lines_fld);//特徴線検索
	for(int i = 0; i < lines_fld.size(); i++)
	{
		//cv::line(img_dst,cv::Point(lines[i][0],lines[i][1]),cv::Point(lines[i][2],lines[i][3]),cv::Scalar(0,0,255), 4, cv::LINE_AA);  
		cv::line(img_fld,cv::Point(lines_fld[i][0],lines_fld[i][1]),cv::Point(lines_fld[i][2],lines_fld[i][3]),cv::Scalar(0,0,255), 1.5, cv::LINE_AA); 
	}
	//線検出終了//

	//線の角度を求める
	double lines2[lines_fld.size()][4],theta0,theta90;//初期定義
	float dep1[lines_fld.size()],dep2[lines_fld.size()];//初期定義

	//Y軸との角度(詳しくは2月の研究ノート)
	theta0=M_PI-atan2((200-0),(100-100));//水平(θ=π/2=1.5708)
	theta90=M_PI-atan2((100-100),(200-0));//垂直(θ=π=3.14159)

	//FLD抽出線とY軸との角度を求める+三次元距離データの結合
	//std::cout <<"並び替え前"<<std::endl;
	int ksk=0;
	for(int i =  0; i < lines_fld.size(); i++)
	{
		dep1[i]= img_depth.at<float>(cv::Point(lines_fld[i][0],lines_fld[i][1]));//点の三次元距離データ取得//20210908ここでDepth情報を取得する必要はない気がする
		dep2[i]= img_depth.at<float>(cv::Point(lines_fld[i][2],lines_fld[i][3]));
		if(dep1[i]>0 && dep2[i]>0) //dep1とdep2が0より大きい時に実行する。(距離データの損失を考慮)
		{
			lines2[ksk][0]=lines_fld[i][0];//ここで距離０を除いてる
			lines2[ksk][1]=lines_fld[i][1];
			lines2[ksk][2]=lines_fld[i][2];
			lines2[ksk][3]=lines_fld[i][3];
			dep1[ksk]=dep1[i];
			dep2[ksk]=dep2[i];
			ksk=ksk+1;//距離データを正しく持つ線のみを取得
		}
	}
	//std::cout <<"テスト:Depthデータが取得可能な線の個数ksk="<<ksk<<std::endl;

	double lines_NNM[lines_fld.size()][4],lines_NNM2[lines_fld.size()][4],lines_NNM_lc[lines_fld.size()],lines_NNM_thetal[lines_fld.size()];
	double lines3[lines_fld.size()][4],lines3_dep[lines_fld.size()][2],lines3_theta[lines_fld.size()];//抽出されたナナメの線以外
	double NNMA[lines_fld.size()],NNMB[lines_fld.size()],NNM_TATE_X[lines_fld.size()][lines_fld.size()],NNM_TATE_Y[lines_fld.size()][lines_fld.size()];
	int lines_NNM_count=0,lines3_count=0;
	double minlength = image.cols * image.cols * 0.02 ;// (線の最小長さの要件を定義する)
	double NNM_A[lines_fld.size()],NNM_C[lines_fld.size()],NNM_A_MAX,NNM_C_MAX;//斜め線の一次関数
	double NNM_XN[lines_fld.size()],NNM_YN[lines_fld.size()],NNM_CHK[lines_fld.size()];//縦線の結合用
	double NNM_L[lines_fld.size()],tempx,tempy;//縦線の長さ
	double NNM_line[lines_fld.size()][4];//縦線結合後
	int nnmno=0;//縦線結合後個数

	//ナナメの線の抽出を行う
	for(int i=0; i<ksk; i++)
	{
		//(ほぼ垂直の場合は無視)
		//ここでは斜めの線と斜めではない線に分類分けを行なっている
		//lines3=斜めではない線,lines_NNM=斜めの線
		if ( abs(lines2[i][0]-lines2[i][2]) < 10 || abs(lines2[i][1]-lines2[i][3]) < 10) //check if almost vertical
		{ 
			lines3[lines3_count][0]=lines2[i][0];//ナナメの線以外を抽出する
			lines3[lines3_count][1]=lines2[i][1];
			lines3[lines3_count][2]=lines2[i][2];
			lines3[lines3_count][3]=lines2[i][3];
			lines3_dep[lines3_count][0]=dep1[ksk];
			lines3_dep[lines3_count][1]=dep2[ksk];
			cv::line(img_FLD_TY,cv::Point(lines3[lines3_count][0],lines3[lines3_count][1]),cv::Point(lines3[lines3_count][2],lines3[lines3_count][3]),cv::Scalar(0,0,0), 2, cv::LINE_AA);

			//FLD抽出線のy軸との角度を求める
			lines3_theta[lines3_count]=M_PI-atan2((lines3[lines3_count][2]-lines3[lines3_count][0]),(lines3[lines3_count][3]-lines3[lines3_count][1]));
			//std::cout <<"FLDの線の傾きlines3_theta["<<lines3_count<<"]("<<lines3_theta[lines3_count]<<")"<< std::endl;
			lines3_count=lines3_count+1;//ナナメの線以外を線を数える
			continue;
		}
		//(短い線を無視する (x1-x2)^2 + (y2-y1)^2 < minlength)
		if( ((lines2[i][0]-lines2[i][2])*(lines2[i][0]-lines2[i][2]) +(lines2[i][1]-lines2[i][3])*(lines2[i][1]-lines2[i][3])) < minlength) continue;
		
		lines_NNM[lines_NNM_count][0]=lines2[i][0];//lines_NNM抽出した斜めの線
		lines_NNM[lines_NNM_count][1]=lines2[i][1];
		lines_NNM[lines_NNM_count][2]=lines2[i][2];
		lines_NNM[lines_NNM_count][3]=lines2[i][3];

		cv::line(img_line2,cv::Point(lines_NNM[lines_NNM_count][0],lines_NNM[lines_NNM_count][1]),cv::Point(lines_NNM[lines_NNM_count][2],lines_NNM[lines_NNM_count][3]),cv::Scalar(0,0,255), 2, cv::LINE_AA); 
		//cv::line(img_line4,cv::Point(lines_NNM[lines_NNM_count][0],lines_NNM[lines_NNM_count][1]),cv::Point(lines_NNM[lines_NNM_count][2],lines_NNM[lines_NNM_count][3]),cv::Scalar(0,0,255), 2, cv::LINE_AA); 
		cv::line(img_dst,cv::Point(lines_NNM[lines_NNM_count][0],lines_NNM[lines_NNM_count][1]),cv::Point(lines_NNM[lines_NNM_count][2],lines_NNM[lines_NNM_count][3]),cv::Scalar(0,255,0), 4, cv::LINE_AA); 

		//座標から一次関数を引く関数
		lines_NNM_thetal[lines_NNM_count]=(M_PI/2)-(M_PI-atan2((lines_NNM[lines_NNM_count][2]-lines_NNM[lines_NNM_count][0]),(lines_NNM[lines_NNM_count][3]-lines_NNM[lines_NNM_count][1])));
		lines_NNM_lc[lines_NNM_count]=(lines_NNM[lines_NNM_count][2]-lines_NNM[lines_NNM_count][0])*(lines_NNM[lines_NNM_count][2]-lines_NNM[lines_NNM_count][0])+(lines_NNM[lines_NNM_count][3]-lines_NNM[lines_NNM_count][1])*(lines_NNM[lines_NNM_count][3]-lines_NNM[lines_NNM_count][1]);
		lines_NNM2[lines_NNM_count][0]=lines_NNM[lines_NNM_count][0]+(cos(-lines_NNM_thetal[lines_NNM_count])*sqrt(lines_NNM_lc[lines_NNM_count]))*1000;//X1座標
		lines_NNM2[lines_NNM_count][1]=lines_NNM[lines_NNM_count][1]+(sin(-lines_NNM_thetal[lines_NNM_count])*sqrt(lines_NNM_lc[lines_NNM_count]))*1000;//Y1座標
		lines_NNM2[lines_NNM_count][2]=lines_NNM[lines_NNM_count][0]+(cos(-lines_NNM_thetal[lines_NNM_count])*sqrt(lines_NNM_lc[lines_NNM_count]))*-1000;//X2座標
		lines_NNM2[lines_NNM_count][3]=lines_NNM[lines_NNM_count][1]+(sin(-lines_NNM_thetal[lines_NNM_count])*sqrt(lines_NNM_lc[lines_NNM_count]))*-1000;//Y2座標

		cv::line(img_line2,cv::Point(lines_NNM2[lines_NNM_count][0],lines_NNM2[lines_NNM_count][1]),cv::Point(lines_NNM2[lines_NNM_count][2],lines_NNM2[lines_NNM_count][3]),cv::Scalar(0,255,0), 1, cv::LINE_AA);
		cv::line(img_line4,cv::Point(lines_NNM2[lines_NNM_count][0],lines_NNM2[lines_NNM_count][1]),cv::Point(lines_NNM2[lines_NNM_count][2],lines_NNM2[lines_NNM_count][3]),cv::Scalar(0,255,0), 1, cv::LINE_AA);
		cv::line(img_FLD_TY,cv::Point(lines_NNM2[lines_NNM_count][0],lines_NNM2[lines_NNM_count][1]),cv::Point(lines_NNM2[lines_NNM_count][2],lines_NNM2[lines_NNM_count][3]),cv::Scalar(0,255,0), 1, cv::LINE_AA);
		cv::line(img_dst,cv::Point(lines_NNM2[lines_NNM_count][0],lines_NNM2[lines_NNM_count][1]),cv::Point(lines_NNM2[lines_NNM_count][2],lines_NNM2[lines_NNM_count][3]),cv::Scalar(0,255,0), 1, cv::LINE_AA);
		NNM_CHK[lines_NNM_count]=0;//斜め線判別要素初期化（斜め線の結合に使用)
		lines_NNM_count=lines_NNM_count+1;//ナナメの線の数をカウント
	}//(ナナメ線抽出)

	//縦線と横線の処理
	//thetaの数値を小さいにソート
	double tmp=0,tmp1x=0,tmp1y=0,tmp2x=0,tmp2y=0,tmpdep1=0,tmpdep2=0,tmp3=0;
	int yokot,tatet,p,yokoyouso,tateyouso;
	double lines3X,lines3Y,yokolines3[lines_fld.size()][5],tatelines3[lines_fld.size()][5],yokotheta[lines_fld.size()],tatetheta[lines_fld.size()];
	double yokoZ1[lines_fld.size()],yokoZ2[lines_fld.size()],tateZ1[lines_fld.size()],tateZ2[lines_fld.size()];
	double yokothetal[lines_fld.size()],tatethetal[lines_fld.size()],yokolc[lines_fld.size()],tatelc[lines_fld.size()],yokol[lines_fld.size()][4],tatel[lines_fld.size()][4];
	double datat[lines_fld.size()],datay[lines_fld.size()];//ヒストグラムデータ用
	int clusCY,clusCT,clusy[lines_fld.size()],clust[lines_fld.size()],MAXY=0,MAXT=0;//クラスタリング用
	double yokothetaCL[lines_fld.size()],tatethetaCL[lines_fld.size()],clusR;//クラスタリング用
	double yokoclusy[100][200][5],tateclust[100][200][5];
	double Average_yoko_theta[lines_fld.size()];
	double MINI_theta=100,mini_theta;
	int CLUSTER[lines_fld.size()],clus_no=0,YOKO_CLUST;
	int nanamet=0,nanamey=0,NANAME_Line[100];
	double lines_T_theta[lines_fld.size()];
	double tateA[lines_fld.size()],tateB[lines_fld.size()];
	double TATE_A[lines_fld.size()],TATE_C[lines_fld.size()],TATE_A_MAX,TATE_C_MAX;//縦線の一次関数y=TATE_Ax+TATE_C
	double TATE_D[lines_fld.size()][lines_fld.size()],TATE_XN[lines_fld.size()],TATE_YN[lines_fld.size()];//縦線の結合用
	double TATE_L[lines_fld.size()],TATE_K[lines_fld.size()];//縦線の長さ
	double TATE_line[lines_fld.size()][4];//縦線結合後
	int tateno=0,tateok=0;//縦線結合後個数
	
	yokot=0,tatet=0,p=0,yokoyouso=0,tateyouso=0;

	//縦線と横線の分類分け
	for (int j=0; j< lines3_count; ++j) 
	{
		lines3X=abs(lines3[j][0]-lines3[j][2]);//傾きを調べる（x成分)
		lines3Y=abs(lines3[j][1]-lines3[j][3]);//傾きを調べる（y成分)
		
		//横線に分類
		if(lines3X>lines3Y)
		{
			//std::cout <<"yoko(lines3X>lines3Y)="<<lines3X<<">"<<lines3Y<< std::endl;
			//std::cout <<"lines3_theta["<<j<<"](ヨコ)="<<lines3_theta[j]<< std::endl;

			yokolines3[yokoyouso][0]=lines3[j][0];//(x成分)
			yokolines3[yokoyouso][1]=lines3[j][1];//(y成分)
			yokolines3[yokoyouso][2]=lines3[j][2];//(x成分)
			yokolines3[yokoyouso][3]=lines3[j][3];//(y成分)
			yokolines3[yokoyouso][4]=lines3_theta[j];

			//yokotheta[yokoyouso]=theta[j];
			yokoZ1[yokoyouso]=lines3_dep[j][0];
			yokoZ2[yokoyouso]=lines3_dep[j][1];
			//cv::line(img_line3,cv::Point(yokolines3[yokoyouso][0],yokolines3[yokoyouso][1]),cv::Point(yokolines3[yokoyouso][2],yokolines3[yokoyouso][3]),cv::Scalar(255,0,0), 2, cv::LINE_AA);
			//cv::line(img_FLD_Y,cv::Point(yokolines3[yokoyouso][0],yokolines3[yokoyouso][1]),cv::Point(yokolines3[yokoyouso][2],yokolines3[yokoyouso][3]),cv::Scalar(255,0,0), 2, cv::LINE_AA);
			cv::line(img_line2,cv::Point(yokolines3[yokoyouso][0],yokolines3[yokoyouso][1]),cv::Point(yokolines3[yokoyouso][2],yokolines3[yokoyouso][3]),cv::Scalar(255,0,0), 2, cv::LINE_AA);
			//cv::circle(img_dst, cv::Point(yokolines3[yokoyouso][0],yokolines3[yokoyouso][1]), 4, cv::Scalar(255, 0, 255), 1.5);
			//cv::circle(img_dst, cv::Point(yokolines3[yokoyouso][2],yokolines3[yokoyouso][3]), 4, cv::Scalar(255, 0, 255), 1.5);

			yokotheta[yokoyouso]=lines3_theta[j]*180/M_PI;//deg表示化
			//θの範囲を0〜180にする
			if(yokotheta[yokoyouso]>=180)
			{
				yokotheta[yokoyouso]=yokotheta[yokoyouso]-180;
				yokolines3[yokoyouso][4]=yokolines3[yokoyouso][4]-M_PI;
			}
			//std::cout <<"yokotheta["<<yokoyouso<<"]="<<yokotheta[yokoyouso]<< std::endl;

			if(yokotheta[yokoyouso]>=90)
			{
				yokotheta[yokoyouso]=180-yokotheta[yokoyouso];
			}
			else
			{
				yokotheta[yokoyouso]=yokotheta[yokoyouso];
			}
			//std::cout <<"yokotheta["<<yokoyouso<<"](クラスタリング用)="<<yokotheta[yokoyouso]<< std::endl;

			clusy[yokoyouso]=0;//クラスタリング用
			yokoyouso=yokoyouso+1;//横線に分類されたグループ数(yokoyouso)
		}
		else //縦線に分類
		{
			if(lines3Y>lines3X)
			{
				//std::cout <<"tate(lines3Y>lines3X)="<<lines3Y<<">"<<lines3X<< std::endl;
				//std::cout <<"lines3_theta["<<j<<"](タテ)="<<lines3_theta[j]<< std::endl;
				tatelines3[tateyouso][0]=lines3[j][0];
				tatelines3[tateyouso][1]=lines3[j][1];
				tatelines3[tateyouso][2]=lines3[j][2];
				tatelines3[tateyouso][3]=lines3[j][3];
				tatelines3[tateyouso][4]=lines3_theta[j];

				//tatetheta[tateyouso]=lines3_theta[j];
				tateZ1[tateyouso]=lines3_dep[j][0];
				tateZ2[tateyouso]=lines3_dep[j][1];
				//cv::line(img_line3,cv::Point(tatelines3[tateyouso][0], tatelines3[tateyouso][1]),cv::Point(tatelines3[tateyouso][2],tatelines3[tateyouso][3]),cv::Scalar(0,0,255), 2, cv::LINE_AA);
				//cv::line(img_FLD_T,cv::Point(tatelines3[tateyouso][0], tatelines3[tateyouso][1]),cv::Point(tatelines3[tateyouso][2],tatelines3[tateyouso][3]),cv::Scalar(0,0,255), 2, cv::LINE_AA);
				cv::line(img_line2,cv::Point(tatelines3[tateyouso][0], tatelines3[tateyouso][1]),cv::Point(tatelines3[tateyouso][2],tatelines3[tateyouso][3]),cv::Scalar(0,0,255), 2, cv::LINE_AA);

				//確率ハフ変換を使用しない時
				tatetheta[tateyouso]=lines3_theta[j]*180/M_PI;//deg表示化

				//θの範囲を0〜180にする
				if(tatetheta[tateyouso]>=180)
				{
					tatetheta[tateyouso]=tatetheta[tateyouso]-180;
					tatelines3[tateyouso][4]=tatelines3[tateyouso][4]-M_PI;
				}
				//std::cout <<"tatetheta["<<tateyouso<<"]="<<tatetheta[tateyouso]<< std::endl;

				//クラスタリング時に最大個数を持つ縦線のクラスタが２つ存在してしまうため、90度で反転させてクラスタリング処理を行う。
				//例(θ=0~10,170~180→180-170=10)
				if(tatetheta[tateyouso]>=90)
				{
					tatetheta[tateyouso]=180-tatetheta[tateyouso];
				}
				else
				{
					tatetheta[tateyouso]=tatetheta[tateyouso];
				}
				//std::cout <<"tatetheta["<<tateyouso<<"](クラスタリング用)="<<tatetheta[tateyouso]<< std::endl;

				clust[tateyouso]=0;//クラスタリング用
				tateyouso=tateyouso+1;//縦線に分類されたグループ数(tateyouso)
			}
		}
	}

	//ここの並び替え上の並び替えとまとめられそう（先に範囲を狭めてから並び替えして分類する感じにしたらできそう）
	//今は並び替えして分類して範囲狭めて再び並び替えしてる
	//クラスタリング用θで並び替えを行う
	//tatethetaの数値を小さいにソート
	tmp=0,tmp1x=0,tmp1y=0,tmp2x=0,tmp2y=0,tmpdep1=0,tmpdep2=0,tmp3=0;
	for (int i=0; i<tateyouso; ++i) 
	{
		for (int j=i+1;j<tateyouso; ++j) 
		{
			if (tatetheta[i] > tatetheta[j]) 
			{
				tmp =  tatetheta[i];
				tmp1x =  tatelines3[i][0];
				tmp1y =  tatelines3[i][1];
				tmp2x =  tatelines3[i][2];
				tmp2y =  tatelines3[i][3];
				tmp3 =  tatelines3[i][4];

				tatetheta[i] = tatetheta[j];
				tatelines3[i][0] = tatelines3[j][0];
				tatelines3[i][1] = tatelines3[j][1];
				tatelines3[i][2] = tatelines3[j][2];
				tatelines3[i][3] = tatelines3[j][3];
				tatelines3[i][4] = tatelines3[j][4];

				tatetheta[j] = tmp;
				tatelines3[j][0] = tmp1x;
				tatelines3[j][1] = tmp1y;
				tatelines3[j][2] = tmp2x;
				tatelines3[j][3] = tmp2y;
				tatelines3[j][4] = tmp3;
			}
		}
	}
	
	//クラスタリング用θで並び替えを行う
	//yokothetaの数値を小さいにソート
	tmp=0,tmp1x=0,tmp1y=0,tmp2x=0,tmp2y=0,tmpdep1=0,tmpdep2=0,tmp3=0;
	for (int i=0; i<yokoyouso; ++i) 
	{
		for (int j=i+1;j<yokoyouso; ++j) 
		{
			if (yokotheta[i] > yokotheta[j]) 
			{
				tmp =  yokotheta[i];
				tmp1x =  yokolines3[i][0];
				tmp1y =  yokolines3[i][1];
				tmp2x =  yokolines3[i][2];
				tmp2y =  yokolines3[i][3];
				tmp3 =  yokolines3[i][4];

				yokotheta[i] = yokotheta[j];
				yokolines3[i][0] = yokolines3[j][0];
				yokolines3[i][1] = yokolines3[j][1];
				yokolines3[i][2] = yokolines3[j][2];
				yokolines3[i][3] = yokolines3[j][3];
				yokolines3[i][4] = yokolines3[j][4];

				yokotheta[j] = tmp;
				yokolines3[j][0] = tmp1x;
				yokolines3[j][1] = tmp1y;
				yokolines3[j][2] = tmp2x;
				yokolines3[j][3] = tmp2y;
				yokolines3[j][4] = tmp3;
			}
		}
	}

	//縦線のクラスタリング
	//タテ線の平行線のクラスタリング(clusRがクラスタリング半径,clusCT:クラスタ数,Clust[]:クラスタ内の個数)
	//最も個数の多い並行線クラスタを各線の並行線とする
	//clusR=0.15,clusCT=0,Average_tate_theta=0;
	clusR=0.5,clusCT=0,Average_tate_theta=0;
	//要素数が0の時は実行しない
	if(tateyouso==0)
	{
		//std::cout <<"実行しない--------------tateyouso="<<tateyouso<< std::endl;
		tate_histgram = false;//タテ線のヒストグラムを実行しない
	}
	else
	{
		tate_histgram = true;//タテ線のヒストグラムを実行
		for (int j=0; j< tateyouso; ++j) 
		{
			if(tateyouso==1)	//要素がひとつしかないとき比較ができない
			{
				//std::cout <<"要素がひとつしかない"<< std::endl;//クラスタ数
				//std::cout <<"クラスタ番号clusCT="<<clusCT<< std::endl;//クラスタ数
				//std::cout <<"abs(tatetheta["<<j<<"]*3-tatetheta["<<j+1<<"]*3)="<<abs(tatetheta[j]*3-tatetheta[j+1]*3)<< std::endl;
				cv::circle(img_graph, cv::Point(tatetheta[j]*6, 240), 3, Scalar(0,5*clusCT,255), -1, cv::LINE_AA);//線の角度グラフ

				// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
				tateclust[0][clust[0]][0]=tatelines3[j][0];//(x成分)
				tateclust[0][clust[0]][1]=tatelines3[j][1];//(y成分)
				tateclust[0][clust[0]][2]=tatelines3[j][2];//(x成分)
				tateclust[0][clust[0]][3]=tatelines3[j][3];//(y成分)
				tateclust[0][clust[0]][4]=tatelines3[j][4];//角度
				clust[MAXT]=1;
			}
			else
			{
				cv::circle(img_graph, cv::Point(tatetheta[j]*6, 240), 3, Scalar(0,5*clusCT,255), -1, cv::LINE_AA);//線の角度グラフ
				if(j==0)	//初回動作時
				{
					//std::cout <<"abs(tatetheta["<<j<<"]*3-tatetheta["<<j+1<<"]*3)="<<abs(tatetheta[j]*3-tatetheta[j+1]*3)<< std::endl;
					if(clusR>abs(tatetheta[j]*3-tatetheta[j+1]*3))	//クラスタリング範囲内
					{
						//std::cout <<"初回動作範囲内j="<<j<<",clusCT="<<clusCT<<",clust[clusCT]="<<clust[clusCT]<< std::endl;
						// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
						tateclust[clusCT][clust[clusCT]][0]=tatelines3[j][0];//(x成分)
						tateclust[clusCT][clust[clusCT]][1]=tatelines3[j][1];//(y成分)
						tateclust[clusCT][clust[clusCT]][2]=tatelines3[j][2];//(x成分)
						tateclust[clusCT][clust[clusCT]][3]=tatelines3[j][3];//(y成分)
						tateclust[clusCT][clust[clusCT]][4]=tatelines3[j][4];//角度

						//cv::line(img_line4,cv::Point(tateclust[clusCT][clust[clusCT]][0],tateclust[clusCT][clust[clusCT]][1]),
						//cv::Point(tateclust[clusCT][clust[clusCT]][2],tateclust[clusCT][clust[clusCT]][3]),cv::Scalar(0,5*clusCT,255), 3, cv::LINE_AA);
						clust[clusCT]=clust[clusCT]+1;//クラスタの内部個数更新
					}
					else	//クラスタリング範囲外
					{
						//std::cout <<"初回動作範囲外j="<<j<<",clusCT="<<clusCT<<"----------------------------"<< std::endl;
						// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
						tateclust[clusCT][clust[clusCT]][0]=tatelines3[j][0];//(x成分)
						tateclust[clusCT][clust[clusCT]][1]=tatelines3[j][1];//(y成分)
						tateclust[clusCT][clust[clusCT]][2]=tatelines3[j][2];//(x成分)
						tateclust[clusCT][clust[clusCT]][3]=tatelines3[j][3];//(y成分)
						tateclust[clusCT][clust[clusCT]][4]=tatelines3[j][4];//角度

						//cv::line(img_line4,cv::Point(tateclust[clusCT][clust[clusCT]][0],tateclust[clusCT][clust[clusCT]][1]),
						//cv::Point(tateclust[clusCT][clust[clusCT]][2],tateclust[clusCT][clust[clusCT]][3]),cv::Scalar(0,5*clusCT,255), 3, cv::LINE_AA);
						MAXT=clusCT;
						clusCT=clusCT+1;//クラスタ更新
					}
				}
				
				if(j!=0&&j+1!=tateyouso)	//中間動作
				{
					//std::cout <<"abs(tatetheta["<<j<<"]*3-tatetheta["<<j+1<<"]*3)="<<abs(tatetheta[j]*3-tatetheta[j+1]*3)<< std::endl;
					//後方クラスタリング半径範囲内
					if(clusR>abs(tatetheta[j]*3-tatetheta[j+1]*3))
					{
						//std::cout <<"中間動作範囲内j="<<j<<",clusCT="<<clusCT<<",clust[clusCT]="<<clust[clusCT]<< std::endl;
						// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
						tateclust[clusCT][clust[clusCT]][0]=tatelines3[j][0];//(x成分)
						tateclust[clusCT][clust[clusCT]][1]=tatelines3[j][1];//(y成分)
						tateclust[clusCT][clust[clusCT]][2]=tatelines3[j][2];//(x成分)
						tateclust[clusCT][clust[clusCT]][3]=tatelines3[j][3];//(y成分)
						tateclust[clusCT][clust[clusCT]][4]=tatelines3[j][4];//角度

						//cv::line(img_line4,cv::Point(tateclust[clusCT][clust[clusCT]][0],tateclust[clusCT][clust[clusCT]][1]),
						//cv::Point(tateclust[clusCT][clust[clusCT]][2],tateclust[clusCT][clust[clusCT]][3]),cv::Scalar(0,5*clusCT,255), 3, cv::LINE_AA);
						clust[clusCT]=clust[clusCT]+1;//クラスタの内部個数更新
					}
					else	//後方クラスタリング半径範囲外
					{
						//std::cout <<"中間動作範囲外j="<<j<<",clusCT="<<clusCT<<"--------------------------------"<< std::endl;
						// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
						tateclust[clusCT][clust[clusCT]][0]=tatelines3[j][0];//(x成分)
						tateclust[clusCT][clust[clusCT]][1]=tatelines3[j][1];//(y成分)
						tateclust[clusCT][clust[clusCT]][2]=tatelines3[j][2];//(x成分)
						tateclust[clusCT][clust[clusCT]][3]=tatelines3[j][3];//(y成分)
						tateclust[clusCT][clust[clusCT]][4]=tatelines3[j][4];//角度

						//cv::line(img_line4,cv::Point(tateclust[clusCT][clust[clusCT]][0],tateclust[clusCT][clust[clusCT]][1]),
						//cv::Point(tateclust[clusCT][clust[clusCT]][2],tateclust[clusCT][clust[clusCT]][3]),cv::Scalar(0,5*clusCT,255), 3, cv::LINE_AA);
						clusCT=clusCT+1;//クラスタ更新
					}
				}
				
				if(j+1==tateyouso)	//最終動作
				{
					//std::cout <<"最終動作j="<<j<<",clusCT="<<clusCT<<",clust[clusCT]="<<clust[clusCT]<< std::endl;
					// クラスタリングされたヨコ線(クラスタ番号,クラスタ内部個数,成分番号)
					tateclust[clusCT][clust[clusCT]][0]=tatelines3[j][0];//(x成分)
					tateclust[clusCT][clust[clusCT]][1]=tatelines3[j][1];//(y成分)
					tateclust[clusCT][clust[clusCT]][2]=tatelines3[j][2];//(x成分)
					tateclust[clusCT][clust[clusCT]][3]=tatelines3[j][3];//(y成分)
					tateclust[clusCT][clust[clusCT]][4]=tatelines3[j][4];//角度

					//cv::line(img_line4,cv::Point(tateclust[clusCT][clust[clusCT]][0],tateclust[clusCT][clust[clusCT]][1]),
					//cv::Point(tateclust[clusCT][clust[clusCT]][2],tateclust[clusCT][clust[clusCT]][3]),cv::Scalar(0,5*clusCT,255), 3, cv::LINE_AA);
					//std::cout <<"最終:最大クラスタMAXT=clusCT="<<MAXT<<",最大クラスタの内部個数clust[MAXT]="<<clust[MAXT]<< std::endl;
				}
				if(clust[clusCT]>=clust[MAXT]) MAXT=clusCT;//最大クラスタをキープする
				//std::cout <<"最大クラスタMAXT=clusCT="<<MAXT<<",最大クラスタの内部個数clust[MAXT]="<<clust[MAXT]<<"\n"<< std::endl;
			}
		}
		//線のグループ化
		//短い線はノイズなので除去する
		for (int i=0; i<clust[MAXT]; ++i) 
		{
			if(sqrt((tateclust[MAXT][i][0]-tateclust[MAXT][i][2])*(tateclust[MAXT][i][0]-tateclust[MAXT][i][2])
			+(tateclust[MAXT][i][1]-tateclust[MAXT][i][3])*(tateclust[MAXT][i][1]-tateclust[MAXT][i][3]))>40)
			{
				tateclust[MAXT][tateok][0]=tateclust[MAXT][i][0];
				tateclust[MAXT][tateok][1]=tateclust[MAXT][i][1];
				tateclust[MAXT][tateok][2]=tateclust[MAXT][i][2];
				tateclust[MAXT][tateok][3]=tateclust[MAXT][i][3];
				tateok=tateok+1;//縦線の合計個数(まとめ後)
			}
		}
		//Xの値が小さい順に並び変える
		tmp=0,tmp1x=0,tmp1y=0,tmp2x=0,tmp2y=0,tmpdep1=0,tmpdep2=0,tmp3=0;
		for (int i=0; i<tateok; ++i) 
		{
			for (int j=i+1;j<tateok; ++j) 
			{
				if (tateclust[MAXT][i][0]+tateclust[MAXT][i][2] > tateclust[MAXT][j][0]+tateclust[MAXT][j][2]) 
				{
					tmp1x =  tateclust[MAXT][i][0];
					tmp1y =  tateclust[MAXT][i][1];
					tmp2x =  tateclust[MAXT][i][2];
					tmp2y =  tateclust[MAXT][i][3];
					tmp3 =  tateclust[MAXT][i][4];

					tateclust[MAXT][i][0] = tateclust[MAXT][j][0];
					tateclust[MAXT][i][1] = tateclust[MAXT][j][1];
					tateclust[MAXT][i][2] = tateclust[MAXT][j][2];
					tateclust[MAXT][i][3] = tateclust[MAXT][j][3];
					tateclust[MAXT][i][4] = tateclust[MAXT][j][4];

					tateclust[MAXT][j][0] = tmp1x;
					tateclust[MAXT][j][1] = tmp1y;
					tateclust[MAXT][j][2] = tmp2x;
					tateclust[MAXT][j][3] = tmp2y;
					tateclust[MAXT][j][4] = tmp3;
				}
			}
			//tateclust[MAXT][j][1]の位置を上にする
			if(tateclust[MAXT][i][1]>tateclust[MAXT][i][3])
			{
				tempy=tateclust[MAXT][i][1];
				tempx=tateclust[MAXT][i][0];
				tateclust[MAXT][i][1]=tateclust[MAXT][i][3];
				tateclust[MAXT][i][0]=tateclust[MAXT][i][2];
				tateclust[MAXT][i][3]=tempy;
				tateclust[MAXT][i][2]=tempx;
			}
			TATE_L[i]=sqrt((tateclust[MAXT][i][0]-tateclust[MAXT][i][2])*(tateclust[MAXT][i][0]-tateclust[MAXT][i][2])
			+(tateclust[MAXT][i][1]-tateclust[MAXT][i][3])*(tateclust[MAXT][i][1]-tateclust[MAXT][i][3]));//縦線の点間の長さ
			TATE_XN[i]=(tateclust[MAXT][i][0]+tateclust[MAXT][i][2])/2;//縦線の中点N
			TATE_YN[i]=(tateclust[MAXT][i][1]+tateclust[MAXT][i][3])/2;
		}

		double clusTATE_R=10;//しきい値
		//縦線の中点と直線の距離を使用し、直線をまとめる(距離がしきい値以内なら同じ線とみなす)
		for (int j=0; j<tateok; ++j) 
		{
			int equal=0;//同じ線の要素数
			TATE_line[tateno][0]=tateclust[MAXT][j][0],TATE_line[tateno][1]=tateclust[MAXT][j][1];//基準はjの線(更新が起きなければj単体)
			TATE_line[tateno][2]=tateclust[MAXT][j][2],TATE_line[tateno][3]=tateclust[MAXT][j][3];
			double TATE_MAX_LONG=TATE_L[j];
			double TATE_MIN=tateclust[MAXT][j][1];
			double TATE_MAX=tateclust[MAXT][j][3];

			//縦線がy軸と平行でない時のみ実行
			if(tateclust[MAXT][j][0]!=tateclust[MAXT][j][2])
			{
				//std::cout <<"縦線が斜めのとき:tatethetal["<<j<<"]"<< std::endl;

				TATE_A[j]=(tateclust[MAXT][j][1]-tateclust[MAXT][j][3])/(tateclust[MAXT][j][0]-tateclust[MAXT][j][2]);
				TATE_C[j]=tateclust[MAXT][j][1]-(((tateclust[MAXT][j][1]-tateclust[MAXT][j][3])/(tateclust[MAXT][j][0]-tateclust[MAXT][j][2]))*tateclust[MAXT][j][0]);
				for (int k=j+1; k<tateok; ++k) 
				{
					TATE_D[j][k]=abs((TATE_A[j]*TATE_XN[k])-TATE_YN[k]+TATE_C[j])/sqrt((TATE_A[j]*TATE_A[j])+1);//点と直線の距離
					//距離がしきい値以下の時同じ線とみなす
					if(TATE_D[j][k]<clusTATE_R)
					{
						//std::cout <<"(斜め)同じ線とみなすTATE_D["<<j<<"]["<<k<<"]="<<TATE_D[j][k]<< std::endl;
						//Y最小が直線の下の端点、Y最大が上の端点になる(y軸並行なのでxは長い方の値を使用)
						if(TATE_MIN>tateclust[MAXT][k][1])
						{
							TATE_MIN=tateclust[MAXT][k][1];
							TATE_line[tateno][1]=tateclust[MAXT][k][1];//Y最小が直線の上の端点
						}
						if(TATE_MAX<tateclust[MAXT][k][3])
						{
							TATE_MAX=tateclust[MAXT][k][3];
							TATE_line[tateno][3]=tateclust[MAXT][k][3];//Y最大が下の端点
						}
						//長さが最も長い直線にまとめる(長い方がデータに信頼性がある)
						if(TATE_MAX_LONG<TATE_L[k])
						{
							TATE_MAX_LONG=TATE_L[k];//最大長さ更新
							TATE_A_MAX=(tateclust[MAXT][k][1]-tateclust[MAXT][k][3])/(tateclust[MAXT][k][0]-tateclust[MAXT][k][2]);
							TATE_C_MAX=tateclust[MAXT][k][1]-(((tateclust[MAXT][k][1]-tateclust[MAXT][k][3])/(tateclust[MAXT][k][0]-tateclust[MAXT][k][2]))*tateclust[MAXT][k][0]);
							//最も長い線の一次関数からy最大、最小のxを求める
							TATE_line[tateno][0]=(TATE_line[tateno][1]-TATE_C_MAX)/TATE_A_MAX;
							TATE_line[tateno][2]=(TATE_line[tateno][3]-TATE_C_MAX)/TATE_A_MAX;
						}
						equal=equal+1;//同じ線の要素数
					}
					else
					{
						//連番で見ているのでしきい値を超えた時点で同じ線は無い
						continue;
					}
				}
			}
			else	//縦線がy軸と平行なとき
			{
				//std::cout <<"縦線がY軸と平行:tatethetal["<<j<<"]"<< std::endl;
				for (int k=j+1; k<tateok; ++k) 
				{
					TATE_D[j][k]=abs(tateclust[MAXT][j][0]-TATE_XN[k]);//点と直線の距離(y軸並行)
					//距離がしきい値以下の時同じ線とみなす
					if(TATE_D[j][k]<clusTATE_R)
					{
						//std::cout <<"(平行)同じ線とみなすTATE_D["<<j<<"]["<<k<<"]="<<TATE_D[j][k]<< std::endl;
						//長さが最も長い直線にまとめる(長い方がデータに信頼性がある)
						if(TATE_MAX_LONG<TATE_L[k])
						{
							TATE_MAX_LONG=TATE_L[k];//最大長さ更新
							TATE_line[tateno][0]=tateclust[MAXT][k][0];//最大長さのxを保存
							TATE_line[tateno][2]=tateclust[MAXT][k][2];
							
						}
						//Y最小が直線の下の端点、Y最大が上の端点になる(y軸並行なのでxは長い方の値を使用)
						if(TATE_MIN>tateclust[MAXT][k][1])
						{
							TATE_MIN=tateclust[MAXT][k][1];
							TATE_line[tateno][1]=tateclust[MAXT][k][1];//Y最小が直線の上の端点
						}
						if(TATE_MAX<tateclust[MAXT][k][3])
						{
							TATE_MAX=tateclust[MAXT][k][3];
							TATE_line[tateno][3]=tateclust[MAXT][k][3];//Y最大が下の端点
						}
						equal=equal+1;//同じ線の要素数
					}
					else
					{
						//連番で見ているのでしきい値を超えた時点で同じ線は無い
						continue;
					}
				}
			}
			cv::circle(img_dst2, cv::Point(TATE_line[tateno][0], TATE_line[tateno][1]), 3, Scalar(255,0,0), -1, cv::LINE_AA);//線の角度グラフ
			cv::circle(img_dstfld, cv::Point(TATE_line[tateno][0], TATE_line[tateno][1]), 3, Scalar(255,0,0), -1, cv::LINE_AA);//線の角度グラフ
			cv::circle(img_dst2, cv::Point(TATE_line[tateno][2], TATE_line[tateno][3]), 3, Scalar(0,255,0), -1, cv::LINE_AA);//線の角度グラフ
			cv::circle(img_dstfld, cv::Point(TATE_line[tateno][2], TATE_line[tateno][3]), 3, Scalar(0,255,0), -1, cv::LINE_AA);//線の角度グラフ
			//短い線はノイズなので除去する
			if(sqrt((TATE_line[tateno][0]-TATE_line[tateno][2])*(TATE_line[tateno][0]-TATE_line[tateno][2])
					+(TATE_line[tateno][1]-TATE_line[tateno][3])*(TATE_line[tateno][1]-TATE_line[tateno][3]))>40)
			{
				tateno=tateno+1;//縦線の合計個数(まとめ後)
			}
			j=j+equal;//同じ線とみなされた個数分進む
		}

		tate_point_curr.resize(1000);//配列初期設定(縦線の中点座標)

		//一次関数を描写するプログラム
		for (int j=0; j<tateno; ++j) 
		{
			//座標から一次関数を引く関数
			tatethetal[j]=(M_PI/2)-(M_PI-atan2((TATE_line[j][2]-TATE_line[j][0]),(TATE_line[j][3]-TATE_line[j][1])));
			tatelc[j]=(TATE_line[j][2]-TATE_line[j][0])*(TATE_line[j][2]-TATE_line[j][0])+(TATE_line[j][3]-TATE_line[j][1])*(TATE_line[j][3]-TATE_line[j][1]);
			tatel[j][0]=TATE_line[j][0]+(cos(-tatethetal[j])*sqrt(tatelc[j]))*100;//X1座標
			tatel[j][1]=TATE_line[j][1]+(sin(-tatethetal[j])*sqrt(tatelc[j]))*100;//Y1座標
			tatel[j][2]=TATE_line[j][0]+(cos(-tatethetal[j])*sqrt(tatelc[j]))*-100;//X2座標
			tatel[j][3]=TATE_line[j][1]+(sin(-tatethetal[j])*sqrt(tatelc[j]))*-100;//Y2座標

			cv::line(img_line4,cv::Point(tatel[j][0],tatel[j][1]),cv::Point(tatel[j][2],tatel[j][3]),cv::Scalar(0,0,255), 1, cv::LINE_AA);
			cv::line(img_dst2,cv::Point(tatel[j][0],tatel[j][1]),cv::Point(tatel[j][2],tatel[j][3]),cv::Scalar(0,0,255), 1, cv::LINE_AA);
			cv::line(img_tate,cv::Point(tatel[j][0],tatel[j][1]),cv::Point(tatel[j][2],tatel[j][3]),cv::Scalar(0,0,255), 1, cv::LINE_AA);
			cv::line(img_dstfld,cv::Point(tatel[j][0],tatel[j][1]),cv::Point(tatel[j][2],tatel[j][3]),cv::Scalar(0,0,255), 1, cv::LINE_AA);
			cv::line(img_dst2,cv::Point(TATE_line[j][0],TATE_line[j][1]),cv::Point(TATE_line[j][2],TATE_line[j][3]),cv::Scalar(0,0,255), 4, cv::LINE_AA);
			cv::line(img_tate,cv::Point(TATE_line[j][0],TATE_line[j][1]),cv::Point(TATE_line[j][2],TATE_line[j][3]),cv::Scalar(0,0,255), 4, cv::LINE_AA);
			cv::line(img_line4,cv::Point(TATE_line[j][0],TATE_line[j][1]),cv::Point(TATE_line[j][2],TATE_line[j][3]),cv::Scalar(0,0,255), 4, cv::LINE_AA);
			cv::line(img_dstfld,cv::Point(TATE_line[j][0],TATE_line[j][1]),cv::Point(TATE_line[j][2],TATE_line[j][3]),cv::Scalar(0,0,255), 4, cv::LINE_AA);

			datat[j]=TATE_line[clust[j]][4];
			//最大クラスタ内の平均角度を求める
			Average_tate_theta=Average_tate_theta+datat[j];

			//縦線の中点を求める
			tate_point_curr[j].x=(TATE_line[j][0]+TATE_line[j][2])/2;
			tate_point_curr[j].y=(TATE_line[j][1]+TATE_line[j][3])/2;
			//std::cout<<"tate_point_curr[j].x"<<tate_point_curr[j].x<<std::endl;
			cv::circle(img_dst2, cv::Point(tate_point_curr[j]), 3, Scalar(0,255,255),  -1, cv::LINE_AA);
			cv::circle(img_dstfld, cv::Point(tate_point_curr[j]), 3, Scalar(0,255,255),  -1, cv::LINE_AA);
		}
		tate_point_curr.resize(tateno);//リサイズ(縦線の中点座標)

		//最大クラスタの要素数が１つだけの時を考慮
		if(clust[MAXT]>1)
		{
			Average_tate_theta=Average_tate_theta/(clust[MAXT]-1);	//最大クラスタの要素が２つ以上なら通常の平均計算
		}
		else
		{
			Average_tate_theta=Average_tate_theta;	//最大クラスタの要素数が２つ未満ならその値を平均値とする
		}

		cv::line(img_graph,cv::Point(100,380),cv::Point(100-100*sin(Average_tate_theta),380+100*cos(Average_tate_theta)),cv::Scalar(0,100,255), 3, cv::LINE_AA);
		//std::cout <<"最大クラスタMAXT=clusCT["<<MAXT<<"]内の平均角度="<<Average_tate_theta<<"\n"<< std::endl;
		//cv::line(img_graph,cv::Point(100,380),cv::Point(100-100*sin(Average_tate_theta/(clust[MAXT]-1)),380+100*cos(Average_tate_theta/(clust[MAXT]-1))),cv::Scalar(0,100,255), 3, cv::LINE_AA);
		//std::cout <<"最大クラスタMAXT=clusCT["<<MAXT<<"]内の平均角度="<<Average_tate_theta/(clust[MAXT]-1)<<"\n"<< std::endl;
	}
	//std::cout <<"\n"<< std::endl;
	//線検出プログラム終了

	//テンプレートマッチング
	TPCC.resize(1000);//配列初期設定(tate_point_camera_c)
	DTPC_ok=0;

	//テンプレートマッチングプログラム
	int leftLine=0,rightLine=0;
	//float sum_left=0,sum_right=0;
	for (int aaa = 0; aaa < tate_point_curr.size(); aaa++)
	{
		if (tate_point_curr[aaa].x<324.473&&tate_point_curr[aaa].x>0)//構造線が画面左側の場合
		{
			leftLine++;//画面左側構造線数
			//sum_left=sum_left+tate_point_curr[aaa].x;
		}
		else if (tate_point_curr[aaa].x>=324.473)//構造線が画面右側の場合
		{
			rightLine++;//画面右側構造線数
			//sum_right=sum_right+tate_point_curr[aaa].x;
		}
	}

	//縦線の中点を使用してテンプレートを作成する(線検出の段階ではDepthの不具合などは考慮していないためここで考慮する+クロップ不具合)
	//std::cout <<"tate_point_curr.size()" <<tate_point_curr.size()<<std::endl;
	for (int i=0; i<tate_point_curr.size(); ++i) 
	{
		if(template_size<tate_point_curr[i].y&&tate_point_curr[i].y<480-template_size)
		{
			if(template_size<tate_point_curr[i].x&&tate_point_curr[i].x<640-template_size)
			{
				DTPC[i] = img_depth.at<float>(cv::Point(tate_point_curr[i].x,tate_point_curr[i].y));//depth_tate_point_curr=DTPC
				if (DTPC[i]>0.001&&DTPC[i]<10000)
				{
					TPCC[DTPC_ok].x = -DTPC[i] * ((tate_point_curr[i].x - 324.473) / 615.337)/1000;//メートル表示変換
					TPCC[DTPC_ok].y = -DTPC[i] * ((tate_point_curr[i].y - 241.696) / 615.458)/1000;
					TPCC[DTPC_ok].z = DTPC[i]/1000;
					tate_point_curr[DTPC_ok] = tate_point_curr[i];
				
					cv::circle(img_tate, tate_point_curr[i], 6, Scalar(255,0,0), -1, cv::LINE_AA);
					//テンプレート作成
					cv::Rect roi2(cv::Point(tate_point_curr[i].x-template_size,tate_point_curr[i].y-template_size), cv::Size(template_size*2, template_size*2));//縦線中点を中心とした16☓16pixelの画像を切り取る
					TPCC_Templ[DTPC_ok] = img_src(roi2); // 切り出し画像
					cv::rectangle(img_tate, roi2,cv::Scalar(255, 255, 255), 2);//テンプレート位置
					//cv::rectangle(img_master_temp, cv::Point(tate_point_curr[i].x-template_size,tate_point_curr[i].y+template_size), 
					//cv::Point(tate_point_curr[i].x+template_size,tate_point_curr[i].y-template_size), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);//四角形を描写(白)
					//std::cout <<"縦線中点の画像座標(DTPC_ok)["<<DTPC_ok<<"]="<<tate_point_curr[DTPC_ok]<< std::endl;//縦線中点の座標(範囲制限後)
					DTPC_ok=DTPC_ok+1;//Depth取得可能+テンプレ制限範囲内の個数をカウント
					//std::cout<<"3"<<std::endl;
				}
				//std::cout <<"TPCC" <<TPCC[i] <<std::endl;
			}
		}
	}

	//DTPC_okkk<< DTPC_ok<<"\n";
	tate_point_curr.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_c)
	tate_point_prev.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_c)
	TPCC.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_c)
	TPCP.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_prev)
	Est_tate_point.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_prev)
	Est_tate_pixel.resize(DTPC_ok);//Depth取得数でリサイズ(tate_point_camera_prev)

	reset = false;//if文切り替え
	//std::cout <<"初回検出プログラム終了"<< std::endl;

	MTPC.resize(1000);//Matching_Tate_Pixel_Curr=MTPC定義(ここの大きさを超えるとエラーになる)

	//パーティクルフィルタ初期設定
	if(kaisu==0)
	{
		std::random_device seed;
		std::mt19937 engine(seed());
		std::normal_distribution<> dist1(0,5.46e-3);//0405条件緩和
		std::normal_distribution<> dist2(0,1.13e-3);//0405条件緩和
		std::normal_distribution<> dist3(0,1.72e-2);//0405条件緩和
		std::normal_distribution<> dist4(0,3.15e-3);//0405条件緩和

		for(int y = 0; y < pnum; y++)
		{
			any[y][0]=dist1(engine);
			any[y][1]=dist2(engine);
			any[y][2]=dist3(engine);
			any[y][3]=dist4(engine);
			/*any[y][4]=dist5(engine);
			any[y][5]=dist6(engine);
			any[y][6]=dist7(engine);
			*/
		}
		std::cout <<"初期設定"<< std::endl;
	}

	//テンプレート範囲予測とテンプレートマッチング
	//テンプレートマッチングは1つ前のテンプレートを使用(テンプレート:t+n-1,マッチング画像:t+n)
	if(time0 != false)
	{
		if (reset == false)	//初回以降動作
		{
			//ロボットの動きから特徴点の運動復元を行う(特徴点の状態方程式を使用)
			//std::cout <<"二回目動作:DTPP_ok="<<DTPP_ok<< std::endl;

			matchT_curr=0;
			for(int i=0;i<DTPP_ok;i++)
			{

				Est_tate_point[i].x=-TPCP[i].x+encoder_odometry_.twist.twist.angular.z*realsec*TPCP[i].z+Act_RobotV*sin(-Act_RobotTH)*realsec;
				Est_tate_point[i].y=-TPCP[i].y;
				Est_tate_point[i].z=TPCP[i].z-encoder_odometry_.twist.twist.angular.z*realsec*-TPCP[i].x+Act_RobotV*cos(-Act_RobotTH)*realsec;
				//std::cout <<"Est_tate_point["<<i<<"].x="<<Est_tate_point[i].x<< std::endl;

				Est_tate_pixel[i].x=324.473+(Est_tate_point[i].x/Est_tate_point[i].z)*615.337;
				Est_tate_pixel[i].y=241.696+(Est_tate_point[i].y/Est_tate_point[i].z)*615.458;

				//cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x,Est_tate_pixel[i].y), 6, Scalar(0,255,255), -1, cv::LINE_AA);//一つ前の画像の座標
				cv::line(img_tate,cv::Point(tate_point_prev[i].x,tate_point_prev[i].y),cv::Point(Est_tate_pixel[i].x,Est_tate_pixel[i].y),cv::Scalar(0,0,255), 1, cv::LINE_AA);

				//求めた運動復元結果からテンプレートマッチングの予測範囲を作る(とりあえずタテヨコ2倍)
				//std::cout << "マッチング範囲限定クロッププログラム"<< std::endl;

				//予測範囲が全て画面内の時
				if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)
				{
					//std::cout << "予測範囲が全て画面内の時["<<i<<"]"<< std::endl;
					cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,Est_tate_pixel[i].y-cropx), cv::Size(cropx*2, cropx*2));//線の中点を中心とした線の画像を切り取る
					EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
					cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					//cv::imshow("EST_tate_scope", EST_tate_scope[i]);//黄色の特徴点を中心としたクロップ画像
				}
				else if(0<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<cropx)	//左側
				{
					//左上(xとyどちらもはみでる)
					if(0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)
					{
						//std::cout << "左上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(0,0), cv::Size(Est_tate_pixel[i].x+cropx, Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
					cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
					else if(cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)//左側(xははみ出ない)
					{
						//std::cout << "左側(xははみ出ない)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(0,Est_tate_pixel[i].y-cropy), cv::Size(Est_tate_pixel[i].x+cropx, cropx*2));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
					else if(480-cropy<Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480)	//左下(xとyどちらもはみでる)
					{
						//std::cout << "左下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(0,Est_tate_pixel[i].y-cropy), cv::Size(Est_tate_pixel[i].x+cropx, 480-Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
				}
				else if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)	//上側(yははみ出ない)
				{
					//std::cout << "上側(yははみ出ない)["<<i<<"]"<<std::endl;
					cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,0), cv::Size(cropx*2, Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
					EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
					cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
				}
				else if(640-cropx<Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640)	//右側
				{
					if(0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)	//右上(xとyどちらもはみでる)
					{
						//std::cout << "右上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,0), cv::Size(640-Est_tate_pixel[i].x+cropx, Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
					else if(cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)	//右側(xははみ出ない)
					{
						//std::cout << "右側(xははみ出ない)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,Est_tate_pixel[i].y-cropy), cv::Size(640-Est_tate_pixel[i].x+cropx, cropy*2));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
					else if(480-cropy<Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480)	//右下(xとyどちらもはみでる)
					{
						//std::cout << "右下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,Est_tate_pixel[i].y-cropy), cv::Size(640-Est_tate_pixel[i].x+cropx, 480-Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
					}
				}
				else if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&480-cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<480)	//下側(yははみ出ない)
				{
					//std::cout << "下側(yははみ出ない)["<<i<<"]"<<std::endl;
					cv::Rect roiEST(cv::Point(Est_tate_pixel[i].x-cropx,Est_tate_pixel[i].y-cropy), cv::Size(cropx*2, 480-Est_tate_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
					EST_tate_scope[i] = img_src(roiEST); // 切り出し画像
					cv::rectangle(img_tate, roiEST,cv::Scalar(0, 255, 255), 2);//マッチング予測範囲
				}
				else
				{
					//std::cout << "画面外["<<i<<"]"<< std::endl;
				}

				//予測範囲に対しテンプレートマッチングを行う
				//std::cout << "テンプレートマッチングプログラム["<<i<<"]"<< std::endl;
				TPCP_Templ[i].copyTo(img_template1); // 切り出し画像
				//cv::imshow("win_TPCP_Templ", TPCP_Templ[i]);//黄色の特徴点を中心としたクロップ画像

				cv::Mat img_minmax1;
				// テンプレートマッチング
				cv::matchTemplate(EST_tate_scope[i], img_template1, img_minmax1, cv::TM_CCOEFF_NORMED);//正規化相互相関(ZNCC)
				cv::minMaxLoc(img_minmax1, &min_val1[i], &max_val1[i], &min_pt1[i], &max_pt1[i]);
				if(0.8<max_val1[i])	//最小値がしきい値以下なら表示
				{
					//予測範囲が全て画面内の時
					if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)
					{
						//std::cout << "マッチング:全て画面内の時["<<i<<"]"<< std::endl;
						cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
						cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
						MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
						MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
						MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像(Matching_Tate_Templ_curr=MTTC)
						//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
						matchT_curr=matchT_curr+1;//マッチングの中心座標個数
					}
					else if(0<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<cropx)	//左側
					{
						//左上(xとyどちらもはみでる)
						if(0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)
						{
							//std::cout << "マッチング:左上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
						else if(cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)	//左側(xははみ出ない)
						{
							//std::cout << "マッチング:左側(xははみ出ない)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
						else if(480-cropy<Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480)	//左下(xとyどちらもはみでる)
						{
							//std::cout << "マッチング:左下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
					}
					else if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)	//上側(yははみ出ない)
					{
						//std::cout << "マッチング:上側(yははみ出ない)["<<i<<"]"<<std::endl;
						cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
						cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
						MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
						MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
						MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
						//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
						matchT_curr=matchT_curr+1;//マッチングの中心座標個数
					}
					else if(640-cropx<Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640)	//右側
					{
						if(0<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<cropy)	//右上(xとyどちらもはみでる)
						{
							//std::cout << "マッチング:右上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
						else if(cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480-cropy)	//右側(xははみ出ない)
						{
							//std::cout << "マッチング:右側(xははみ出ない)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
						else if(480-cropy<Est_tate_pixel[i].y&&Est_tate_pixel[i].y<=480)	//右下(xとyどちらもはみでる)
						{
							//std::cout << "マッチング:右下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
							MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
							//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
							matchT_curr=matchT_curr+1;//マッチングの中心座標個数
						}
					}
					else if(cropx<=Est_tate_pixel[i].x&&Est_tate_pixel[i].x<=640-cropx&&480-cropy<=Est_tate_pixel[i].y&&Est_tate_pixel[i].y<480)	//下側(yははみ出ない)
					{
						//std::cout << "マッチング:下側(yははみ出ない)["<<i<<"]"<<std::endl;
						cv::rectangle(img_tate, cv::Rect(Est_tate_pixel[i].x-cropx+max_pt1[i].x, Est_tate_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
						cv::circle(img_tate, cv::Point(Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標
						MTPC[matchT_curr].x=Est_tate_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
						MTPC[matchT_curr].y=Est_tate_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
						MTTC[matchT_curr]=TPCP_Templ[i];//マッチングしたテンプレート画像
						//std::cout <<"マッチングの中心座標[matchT_curr="<<matchT_curr<<"]="<<MTPC[matchT_curr]<< std::endl;
						matchT_curr=matchT_curr+1;//マッチングの中心座標個数
					}
					else
					{
						//std::cout << "画面外["<<i<<"]"<< std::endl;
					}
				}
			}

			MTPC.resize(matchT_curr);//Depth取得可能数でリサイズ(運動復元画像座標)
			MT_curr_pixel.resize(1000);//配列数初期設定(Depth取得可能なマッチング中心画像座標)
			MT_curr_camera.resize(1000);//配列数初期設定(Depth取得可能なマッチング中心カメラ座標)

			//マッチングしたテンプレートをマッチテンプレートとしてキープする
			//マッチテンプレートのDepthが取得不能な点を削除
			DMT_curr_ok=0;
			for (int i = 0; i < matchT_curr; i++) 
			{
				DMT_curr[i] = img_depth.at<float>(cv::Point(MTPC[i].x,MTPC[i].y));//DMT_curr=Depth_MTPC
				
				//std::cout<<"DMT_curr"<<DMT_curr[i]<<std::endl;
				//Depthが取得できない特徴点を削除する+Depthの外れ値を除く
				if(DMT_curr[i]>0.001&&DMT_curr[i]<10000)
				{
					MT_curr_Templ[DMT_curr_ok] = MTTC[i];//Depth取得可能なマッチテンプレート
					MT_curr_pixel[DMT_curr_ok] = MTPC[i];//Depth取得可能なマッチング中心画像座標
					MT_curr_camera[DMT_curr_ok].x = -DMT_curr[i] * ((MTPC[i].x - 324.473) / 615.337)/1000;//カメラ座標変換
					MT_curr_camera[DMT_curr_ok].y = -DMT_curr[i] * ((MTPC[i].y - 241.696) / 615.458)/1000;
					MT_curr_camera[DMT_curr_ok].z = DMT_curr[i]/1000;
					DMT_curr_new[DMT_curr_ok]=DMT_curr[i];

					DMT_curr_ok=DMT_curr_ok+1;//Depthが取得可能なマッチテンプレート数
				}
			}

			//std::cout <<"新規テンプレート数:DMT_curr_ok="<<DMT_curr_ok<< std::endl;
			//類似するテンプレートを削除する
			int notsimil=0;
			for (int i = 0; i < DMT_curr_ok; i++) 
			{
				int similar=0;
				for (int j = i+1; j < DMT_curr_ok; j++) 
				{
					length=sqrt((MT_curr_pixel[i].x-MT_curr_pixel[j].x)*(MT_curr_pixel[i].x-MT_curr_pixel[j].x)
						+(MT_curr_pixel[i].y-MT_curr_pixel[j].y)*(MT_curr_pixel[i].y-MT_curr_pixel[j].y));
					if(template_size*5>length) similar=similar+1;
				}
				//類似するテンプレートを繰り上げ削除
				if(similar==0)
				{
					MT_curr_Templ[notsimil]=MT_curr_Templ[i];
					MT_curr_pixel[notsimil]=MT_curr_pixel[i];
					MT_curr_camera[notsimil]=MT_curr_camera[i];
					DMT_curr_new[notsimil]=DMT_curr_new[i];
					notsimil=notsimil+1;
				}
			}
			DMT_curr_ok=notsimil;
			//std::cout <<"新規テンプレート数(修正後):DMT_curr_ok="<<DMT_curr_ok<< std::endl;

			MT_curr_pixel.resize(DMT_curr_ok);//Depth取得可能数でリサイズ(マッチング中心画像座標)
			MT_curr_camera.resize(DMT_curr_ok);//Depth取得可能数でリサイズ(マッチング中心カメラ座標)

			Est_MT_point.resize(1000);//配列初期設定
			Est_MT_pixel.resize(1000);//配列初期設定

			//世界座標推定(マーカー観測時)
			//テンプレートに一番近いマーカーの座標を使用する(カメラ観測座標で)
			//(マーカー観測が不可能な場合は3回目動作後に推定する)
			MT_curr_world.resize(1000);//初期設定
			if(markerIds.size() > 0)
			{
				//std::cout <<"マーカー観測可能"<< std::endl;
				int minMC;
				for (int i = 0; i < DMT_curr_ok; i++) 
				{
					double minlengh=1000000;
					for (int i = 0; i < markerIds.size(); i++)
					{
						//理論深度値
						depth_ideal[markerIds.at(i)]=-(MarkerW[markerIds.at(i)].at<float>(0)-aggre_RobotY)*sin(-aggre_RobotTH)+(MarkerW[markerIds.at(i)].at<float>(2)-aggre_RobotX)*cos(-aggre_RobotTH);
						yoko_ideal[markerIds.at(i)]=-(MarkerW[markerIds.at(i)].at<float>(0)-aggre_RobotY)*cos(-aggre_RobotTH)-(MarkerW[markerIds.at(i)].at<float>(2)-aggre_RobotX)*sin(-aggre_RobotTH);
							//std::cout<<"理論深度値="<<depth_ideal[markerIds.at(i)]<<std::endl;
						//  //std::cout<<"理論横方向="<<yoko_ideal[markerIds.at(i)]<<std::endl;
					}

					for(int j = 0; j < markerIds.size(); j++)
					{
						length=sqrt((MT_curr_camera[i].x-MC_point[markerIds.at(j)][0])*(MT_curr_camera[i].x-MC_point[markerIds.at(j)][0])
								+(MT_curr_camera[i].y-MC_point[markerIds.at(j)][1])*(MT_curr_camera[i].y-MC_point[markerIds.at(j)][1])
								+(MT_curr_camera[i].z-MC_point[markerIds.at(j)][2])*(MT_curr_camera[i].z-MC_point[markerIds.at(j)][2]));
						length=sqrt((MT_curr_camera[i].x-MC_point[markerIds.at(j)][0])*(MT_curr_camera[i].x-MC_point[markerIds.at(j)][0])
								+(MT_curr_camera[i].z-MC_point[markerIds.at(j)][2])*(MT_curr_camera[i].z-MC_point[markerIds.at(j)][2]));
						//length算出式を変更　縦方向(y軸)を計算に入れない

						if(minlengh>length)
						{
							minMC=j;
							minlengh=length;
						}

					}

					if (MT_curr_camera[i].z>4.0)//補間プログラム（観測距離がしきい値以上の場合）hokann
					{
						MT_curr_world[i].x=MarkerW[markerIds.at(minMC)].at<float>(0)+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*cos(-aggre_RobotTH)-(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*sin(-aggre_RobotTH);
						MT_curr_world[i].y=MarkerW[markerIds.at(minMC)].at<float>(1)+(MT_curr_camera[i].y-MC_point[markerIds.at(minMC)][1]);
						MT_curr_world[i].z=MarkerW[markerIds.at(minMC)].at<float>(2)+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*sin(-aggre_RobotTH)+(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*cos(-aggre_RobotTH);          
					}
					else
					{
						MT_curr_world[i].x=aggre_RobotY+(MT_curr_camera[i].x)*cos(-aggre_RobotTH)-(MT_curr_camera[i].z)*sin(-aggre_RobotTH);
						MT_curr_world[i].y=MarkerW[markerIds.at(minMC)].at<float>(1)+(MT_curr_camera[i].y-MC_point[markerIds.at(minMC)][1]);
						MT_curr_world[i].z=aggre_RobotX+(MT_curr_camera[i].x)*sin(-aggre_RobotTH)+(MT_curr_camera[i].z)*cos(-aggre_RobotTH);
					}
					
					MT_curr_world[i].x=aggre_RobotY+(MT_curr_camera[i].x)*cos(-aggre_RobotTH)-(MT_curr_camera[i].z)*sin(-aggre_RobotTH);
					MT_curr_world[i].y=MarkerW[markerIds.at(minMC)].at<float>(1)+(MT_curr_camera[i].y-MC_point[markerIds.at(minMC)][1]);
					MT_curr_world[i].z=aggre_RobotX+(MT_curr_camera[i].x)*sin(-aggre_RobotTH)+(MT_curr_camera[i].z)*cos(-aggre_RobotTH);

					MT_curr_world[i].x=aggre_RobotY+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*cos(-aggre_RobotTH)-(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*sin(-aggre_RobotTH);
					MT_curr_world[i].y=MarkerW[markerIds.at(minMC)].at<float>(1)+(MT_curr_camera[i].y-MC_point[markerIds.at(minMC)][1]);
					MT_curr_world[i].z=aggre_RobotX+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*sin(-aggre_RobotTH)+(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*cos(-aggre_RobotTH);

					MT_curr_world[i].x=MarkerW[markerIds.at(minMC)].at<float>(0)+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*cos(-aggre_RobotTH)-(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*sin(-aggre_RobotTH);
					MT_curr_world[i].y=MarkerW[markerIds.at(minMC)].at<float>(1)+(MT_curr_camera[i].y-MC_point[markerIds.at(minMC)][1]);
					MT_curr_world[i].z=MarkerW[markerIds.at(minMC)].at<float>(2)+(MT_curr_camera[i].x-MC_point[markerIds.at(minMC)][0])*sin(-aggre_RobotTH)+(MT_curr_camera[i].z-MC_point[markerIds.at(minMC)][2])*cos(-aggre_RobotTH);
				}
				MT_curr_world.resize(DMT_curr_ok);//Depth取得可能数でリサイズ(マッチング中心世界座標)
			}

			//3回目以降動作
			if(Tracking == true)
			{
				//std::cout <<"3回目以降動作"<< std::endl;
				//ロボットの動きから一つ前のマッチング座標の運動復元を行う(特徴点の状態方程式を使用)
				EST_MT_ok=0;
				for(int i=0;i<DMT_prev_ok;i++)
				{

					Est_MT_point[i].x=-MT_prev_camera[i].x+encoder_odometry_.twist.twist.angular.z*realsec*MT_prev_camera[i].z+Act_RobotV*sin(-Act_RobotTH)*realsec;
					Est_MT_point[i].y=-MT_prev_camera[i].y;
					Est_MT_point[i].z=MT_prev_camera[i].z-encoder_odometry_.twist.twist.angular.z*realsec*-MT_prev_camera[i].x+Act_RobotV*cos(-Act_RobotTH)*realsec;

					Est_MT_pixel[i].x=324.473+(Est_MT_point[i].x/Est_MT_point[i].z)*615.337;
					Est_MT_pixel[i].y=241.696+(Est_MT_point[i].y/Est_MT_point[i].z)*615.458;

					//cv::circle(img_tate, cv::Point(Est_pixel[i].x,Est_pixel[i].y), 6, Scalar(0,255,255), -1, cv::LINE_AA);//一つ前の画像の座標
					//一つ前のマッチ画像座標と次の推定画像座標
					cv::line(img_tate,cv::Point(MT_prev_pixel[i].x,MT_prev_pixel[i].y),cv::Point(Est_MT_pixel[i].x,Est_MT_pixel[i].y),cv::Scalar(0,0,255), 1, cv::LINE_AA);//180度(180*3)

					//求めた運動復元結果からテンプレートマッチングの予測範囲を作る(とりあえずタテヨコ2倍)
					//ここで予測点が画面外に行ったらそのテンプレートを削除する
					//std::cout << "マッチング範囲限定クロッププログラム"<< std::endl;
					MTcoix[i]=Est_MT_pixel[i].x+template_size;//予測範囲の中心座標
					MTcoiy[i]=Est_MT_pixel[i].y+template_size;//予測範囲の中心座標
					//予測範囲が全て画面内の時
					if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)
					{
						//std::cout << "予測範囲が全て画面内の時["<<i<<"]"<< std::endl;
						cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,Est_MT_pixel[i].y-cropx), cv::Size(cropx*2, cropx*2));//線の中点を中心とした線の画像を切り取る
						EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
						MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
						MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
						Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
						Est_MT_point[EST_MT_ok] = Est_MT_point[i];
						MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
						MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
						MT_prev_world[EST_MT_ok]=MT_prev_world[i];
						MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
						//cv::imshow("EST_MT_scope", EST_MT_scope[EST_MT_ok]);//黄色の特徴点を中心としたクロップ画像
						EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
					}
					else if(0<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<cropx)	//左側
					{
						//左上(xとyどちらもはみでる)
						if(0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)
						{
							//std::cout << "左上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(0,0), cv::Size(Est_MT_pixel[i].x+cropx, Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
						else if(cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)	//左側(xははみ出ない)
						{
							//std::cout << "左側(xははみ出ない)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(0,Est_MT_pixel[i].y-cropy), cv::Size(Est_MT_pixel[i].x+cropx, cropx*2));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
						else if(480-cropy<Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480)	//左下(xとyどちらもはみでる)
						{
							//std::cout << "左下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(0,Est_MT_pixel[i].y-cropy), cv::Size(Est_MT_pixel[i].x+cropx, 480-Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
					}
					else if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)	//上側(yははみ出ない)
					{
						//std::cout << "上側(yははみ出ない)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,0), cv::Size(cropx*2, Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
						MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
						MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
						Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
						Est_MT_point[EST_MT_ok] = Est_MT_point[i];
						MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
						MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
						MT_prev_world[EST_MT_ok]=MT_prev_world[i];
						MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
						EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
					}
					else if(640-cropx<Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640)	//右側
					{
						//右上(xとyどちらもはみでる)
						if(0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)
						{
							//std::cout << "右上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,0), cv::Size(640-Est_MT_pixel[i].x+cropx, Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
						else if(cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)	//右側(xははみ出ない)
						{
							//std::cout << "右側(xははみ出ない)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,Est_MT_pixel[i].y-cropy), cv::Size(640-Est_MT_pixel[i].x+cropx, cropy*2));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
						else if(480-cropy<Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480)	//右下(xとyどちらもはみでる)
						{
							//std::cout << "右下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
							cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,Est_MT_pixel[i].y-cropy), cv::Size(640-Est_MT_pixel[i].x+cropx, 480-Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
							EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
							MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
							MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
							Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
							Est_MT_point[EST_MT_ok] = Est_MT_point[i];
							MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
							MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
							MT_prev_world[EST_MT_ok]=MT_prev_world[i];
							MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
							cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
							EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
						}
					}
					else if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&480-cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<480)	//下側(yははみ出ない)
					{
						//std::cout << "下側(yははみ出ない)["<<i<<"]"<<std::endl;
						cv::Rect roiEST(cv::Point(Est_MT_pixel[i].x-cropx,Est_MT_pixel[i].y-cropy), cv::Size(cropx*2, 480-Est_MT_pixel[i].y+cropy));//線の中点を中心とした線の画像を切り取る
						EST_MT_scope[EST_MT_ok] = img_src(roiEST); // 切り出し画像
						MTcoix[EST_MT_ok]=Est_MT_pixel[i].x;
						MTcoiy[EST_MT_ok]=Est_MT_pixel[i].y;
						Est_MT_pixel[EST_MT_ok] = Est_MT_pixel[i];
						Est_MT_point[EST_MT_ok] = Est_MT_point[i];
						MT_prev_pixel[EST_MT_ok]=MT_prev_pixel[i];
						MT_prev_camera[EST_MT_ok]=MT_prev_camera[i];
						MT_prev_world[EST_MT_ok]=MT_prev_world[i];
						MT_prev_Templ[EST_MT_ok]=MT_prev_Templ[i];
						cv::rectangle(img_tate, roiEST,cv::Scalar(0, 50, 255), 2);//マッチング予測範囲
						EST_MT_ok=EST_MT_ok+1;//予測範囲が画面内のテンプレート数
					}
					else
					{
						//std::cout << "画面外["<<i<<"]"<< std::endl;
					}
				}//for(int i=0;i<DMT_prev_ok;i++)→end (範囲予測+テンプレートマッチング)
				//std::cout << "EST_MT_ok="<<EST_MT_ok<< std::endl;
				Est_MT_pixel.resize(EST_MT_ok);//リサイズ
				Est_MT_point.resize(EST_MT_ok);//リサイズ
				MT_prev_pixel.resize(EST_MT_ok);//リサイズ
				MT_prev_camera.resize(EST_MT_ok);//リサイズ
				MT_prev_world.resize(EST_MT_ok);//リサイズ
				MTPP.resize(1000);//配列初期設定

				//テンプレートマッチングが原因っぽい
				//予測範囲に対しテンプレートマッチングを行う
				matchT_prev=0;
				for(int i=0;i<EST_MT_ok;i++)
				{
					//std::cout <<"\n"<< std::endl;
					//std::cout << "テンプレートマッチングプログラム["<<i<<"]"<< std::endl;
					//MT_prev_Templ[i].copyTo(img_template1); // 切り出し画像(多分これが原因)
					img_template1 = MT_prev_Templ[i].clone();

					//テンプレートの拡大-----------------------------------------------
					//double scale=sqrt(Est_MT_point[i].x*Est_MT_point[i].x+Est_MT_point[i].y*Est_MT_point[i].y+Est_MT_point[i].z*Est_MT_point[i].z)
					///sqrt(MT_prev_camera[i].x*MT_prev_camera[i].x+MT_prev_camera[i].y*MT_prev_camera[i].y+MT_prev_camera[i].z*MT_prev_camera[i].z);
					//cv::resize(img_template1, img_template1, cv::Size(), scale, scale);

					cv::Mat img_minmax1;
					// テンプレートマッチング
					cv::matchTemplate(EST_MT_scope[i], img_template1, img_minmax1, cv::TM_CCOEFF_NORMED);//正規化相互相関(ZNCC)
					cv::minMaxLoc(img_minmax1, &min_val1[i], &max_val1[i], &min_pt1[i], &max_pt1[i]);
					//std::cout << "min_val1(白)["<<i<<"]=" << min_val1[i] << std::endl;//一致度が上がると値が小さくなる
					//std::cout << "max_val1(白)["<<i<<"]=" << max_val1[i] << std::endl;

					if(0.8<max_val1[i])	//最小値がしきい値以下なら表示
					{
						//予測範囲が全て画面内の時
						if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)
						{
							//std::cout << "マッチング:全て画面内の時["<<i<<"]"<< std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

							MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
							MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

							//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
							MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
							MT_prev_camera[matchT_prev]=MT_prev_camera[i];
							MT_prev_world[matchT_prev]=MT_prev_world[i];
							MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
							//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
							matchT_prev=matchT_prev+1;//マッチングの中心座標個数
						}
						else if(0<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<cropx)	//左側
						{
							//左上(xとyどちらもはみでる)
							if(0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)
							{
								//std::cout << "マッチング:左上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(max_pt1[i].x, max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(max_pt1[i].x+(img_template1.cols/2), max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(max_pt1[i].x, max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
							else if(cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)	//左側(xははみ出ない)
							{
								//std::cout << "マッチング:左側(xははみ出ない)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
							else if(480-cropy<Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480)	//左下(xとyどちらもはみでる)
							{
								//std::cout << "マッチング:左下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
						}
						else if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)	//上側(yははみ出る)
						{
							//std::cout << "マッチング:上側(yははみ出る)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

							MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPP[matchT_prev].y=max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
							MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

							//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
							MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
							MT_prev_camera[matchT_prev]=MT_prev_camera[i];
							MT_prev_world[matchT_prev]=MT_prev_world[i];
							MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
							//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
							matchT_prev=matchT_prev+1;//マッチングの中心座標個数
						}
						else if(640-cropx<Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640)	//右側
						{
							//右上(xとyどちらもはみでる)
							if(0<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<cropy)
							{
								//std::cout << "マッチング:右上(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
							else if(cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480-cropy)	//右側(xははみ出ない)
							{
								//std::cout << "マッチング:右側(xははみ出ない)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
							else if(480-cropy<Est_MT_pixel[i].y&&Est_MT_pixel[i].y<=480)	//右下(xとyどちらもはみでる)
							{
								//std::cout << "マッチング:右下(xとyどちらもはみでる)["<<i<<"]"<<std::endl;
								cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
								cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

								MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
								MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
								cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
								MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

								//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
								MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
								MT_prev_camera[matchT_prev]=MT_prev_camera[i];
								MT_prev_world[matchT_prev]=MT_prev_world[i];
								MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
								//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
								matchT_prev=matchT_prev+1;//マッチングの中心座標個数
							}
						}
						else if(cropx<=Est_MT_pixel[i].x&&Est_MT_pixel[i].x<=640-cropx&&480-cropy<=Est_MT_pixel[i].y&&Est_MT_pixel[i].y<480)	//下側(yははみ出ない)
						{
							//std::cout << "マッチング:下側(yははみ出ない)["<<i<<"]"<<std::endl;
							cv::rectangle(img_tate, cv::Rect(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y, img_template1.cols, img_template1.rows), cv::Scalar(0, 255, 0), 3);//白枠
							cv::circle(img_tate, cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2), Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2)), 5, cv::Scalar(0, 255, 0), -1);//テンプレートの中心座標

							MTPP[matchT_prev].x=Est_MT_pixel[i].x-cropx+max_pt1[i].x+(img_template1.cols/2);//マッチング中心座標(画像全体座標)
							MTPP[matchT_prev].y=Est_MT_pixel[i].y-cropy+max_pt1[i].y+(img_template1.rows/2);//マッチング中心座標(画像全体座標)
							cv::Rect roi_match(cv::Point(Est_MT_pixel[i].x-cropx+max_pt1[i].x, Est_MT_pixel[i].y-cropy+max_pt1[i].y), cv::Size(template_size*2, template_size*2));//テンプレートの更新
							MTTP[matchT_prev] = img_src(roi_match); // 切り出し画像

							//MTTP[matchT_prev]=img_template1;//マッチングしたテンプレート画像
							MT_prev_pixel[matchT_prev]=MT_prev_pixel[i];
							MT_prev_camera[matchT_prev]=MT_prev_camera[i];
							MT_prev_world[matchT_prev]=MT_prev_world[i];
							MT_prev_Templ[matchT_prev]=MT_prev_Templ[i];
							//std::cout <<"マッチングの中心座標[matchT_prev="<<matchT_prev<<"]="<<MTPP[matchT_prev]<< std::endl;
							matchT_prev=matchT_prev+1;//マッチングの中心座標個数
						}
						else
						{
							//std::cout << "画面外["<<i<<"]"<< std::endl;
						}
					}//if(min_val1[i]<max_val1[i]*0.05)→end(テンプレートマッチング)
					else
					{
						//std::cout << "マッチしない["<<i<<"]((0.99<max_val1["<<i<<"])="<<0.99<<"<"<<max_val1[i]<< std::endl;
					}
				}//for(int i=0;i<DMT_prev_ok;i++)→end (範囲予測+テンプレートマッチング)

				//std::cout <<"matchT_prev="<<matchT_prev<< std::endl;
				MTPP.resize(matchT_prev);//Depth取得可能数でリサイズ(運動復元画像座標)
				MT_prev_pixel.resize(matchT_prev);//リサイズ
				MT_prev_camera.resize(matchT_prev);//リサイズ
				MT_prev_world.resize(matchT_prev);//リサイズ
				MT_curr2_pixel.resize(1000);//配列数初期設定(Depth取得可能なマッチング中心画像座標)
				MT_curr2_camera.resize(1000);//配列数初期設定(Depth取得可能なマッチング中心カメラ座標)
				MT_curr2_world.resize(1000);//配列数初期設定(Depth取得可能なマッチング中心カメラ座標)

				//マッチングしたテンプレートをマッチテンプレートとしてキープする
				//マッチテンプレートのDepthが取得不能な点を削除
				DMT_curr2_ok=0;
				for (int i = 0; i < matchT_prev; i++) 
				{
					DMT_curr2[i] = img_depth.at<float>(cv::Point(MTPP[i].x,MTPP[i].y));//DMP_prev=Depth_MTPP
					//std::cout <<"DMT_curr2[i]="<<DMT_curr2[i]<< std::endl;
					//Depthが取得できない特徴点を削除する+Depthの外れ値を除く
					if(DMT_curr2[i]>0.001&&DMT_curr2[i]<10000)	//深度制限10000→6000
					{
						MT_curr2_Templ[DMT_curr2_ok] = MTTP[i];//Depth取得可能なマッチテンプレート
						MT_curr2_pixel[DMT_curr2_ok] = MTPP[i];//Depth取得可能なマッチング中心画像座標
						MT_curr2_camera[DMT_curr2_ok].x = -DMT_curr2[i] * ((MTPP[i].x - 324.473) / 615.337)/1000;//カメラ座標変換
						MT_curr2_camera[DMT_curr2_ok].y = -DMT_curr2[i] * ((MTPP[i].y - 241.696) / 615.458)/1000;
						MT_curr2_camera[DMT_curr2_ok].z = DMT_curr2[i]/1000;              
						MT_curr2_world[DMT_curr2_ok] = MT_prev_world[i];//Depth取得可能なマッチング中心世界座標
						//camera_yoko<< MT_curr2_camera[DMT_curr2_ok].x<<"\n";
						//camera_tate<< MT_curr2_camera[DMT_curr2_ok].y<<"\n";
						//camera_okuyuki<< MT_curr2_camera[DMT_curr2_ok].z<<"\n";

						DMT_curr2_ok=DMT_curr2_ok+1;//Depthが取得可能な全マッチテンプレート数
						//DMT_prev_ok3=DMT_prev_ok3+1;//保持テンプレート数
					}
				}

				//世界座標推定(マーカー観測不能時)
				//マーカーが観測できない場合(観測可能な場合は2回目動作で世界座標を推定する)
				//新規テンプレートに一番近いキープテンプレートの世界座標を使用する(カメラ観測座標で)
				MT_curr_world.resize(1000);//初期設定
				if(markerIds.size() <= 0)
				{
					// std::cout <<"マーカー観測不能時"<< std::endl;
					double minlengh;
					int minMC;
					//std::cout <<"DMT_curr_ok="<<DMT_curr_ok<< std::endl;
					//std::cout <<"DMT_curr2_ok="<<DMT_curr2_ok<< std::endl;
					if(DMT_curr2_ok!=0)
					{
						for (int i = 0; i < DMT_curr_ok; i++) 
						{
							minlengh=1000000;
							for(int j = 0; j < DMT_curr2_ok; j++)
							{
								//length=sqrt((MT_curr_camera[i].x-MT_curr2_camera[j].x)*(MT_curr_camera[i].x-MT_curr2_camera[j].x)
								//+(MT_curr_camera[i].y-MT_curr2_camera[j].y)*(MT_curr_camera[i].y-MT_curr2_camera[j].y)
								//+(MT_curr_camera[i].z-MT_curr2_camera[j].z)*(MT_curr_camera[i].z-MT_curr2_camera[j].z));
								length=sqrt((MT_curr_camera[i].x-MT_curr2_camera[j].x)*(MT_curr_camera[i].x-MT_curr2_camera[j].x)
											+(MT_curr_camera[i].z-MT_curr2_camera[j].z)*(MT_curr_camera[i].z-MT_curr2_camera[j].z));

								if(minlengh>length)
								{
									minMC=j;
									minlengh=length;
								}
							}
							//std::cout <<"DMT_curr_ok="<<DMT_curr_ok<< std::endl;
							//std::cout <<"minMC["<<i<<"]="<<minMC<<std::endl;

							MT_curr_world[i].x=MT_curr2_world[minMC].x+(MT_curr_camera[i].x-MT_curr2_camera[minMC].x)*cos(-aggre_RobotTH)-(MT_curr_camera[i].z-MT_curr2_camera[minMC].z)*sin(-aggre_RobotTH);
							MT_curr_world[i].y=MT_curr2_world[minMC].y+(MT_curr_camera[i].y-MT_curr2_camera[minMC].y);
							MT_curr_world[i].z=MT_curr2_world[minMC].z+(MT_curr_camera[i].x-MT_curr2_camera[minMC].x)*sin(-aggre_RobotTH)+(MT_curr_camera[i].z-MT_curr2_camera[minMC].z)*cos(-aggre_RobotTH);
						}
					}
					else if(DMT_curr2_ok==0)	//一つ前のテンプレートが追跡不能であった場合はマッチング前のテンプレートデータを利用する
					{
						for (int i = 0; i < DMT_curr_ok; i++) 
						{
							minlengh=1000000;
							for(int j = 0; j < EST_MT_ok; j++)
							{
								//length=sqrt((MT_curr_camera[i].x-MT_prev_camera[j].x)*(MT_curr_camera[i].x-MT_prev_camera[j].x)
								//+(MT_curr_camera[i].y-MT_prev_camera[j].y)*(MT_curr_camera[i].y-MT_prev_camera[j].y)
								//+(MT_curr_camera[i].z-MT_prev_camera[j].z)*(MT_curr_camera[i].z-MT_prev_camera[j].z));
								length=sqrt((MT_curr_camera[i].x-MT_prev_camera[j].x)*(MT_curr_camera[i].x-MT_prev_camera[j].x)
											+(MT_curr_camera[i].z-MT_prev_camera[j].z)*(MT_curr_camera[i].z-MT_prev_camera[j].z));

								if(minlengh>length)
								{
									minMC=j;
									minlengh=length;
								}
							}

							MT_curr_world[i].x=MT_prev_world[minMC].x+(MT_curr_camera[i].x-MT_prev_camera[minMC].x)*cos(-aggre_RobotTH)-(MT_curr_camera[i].z-MT_prev_camera[minMC].z)*sin(-aggre_RobotTH);
							MT_curr_world[i].y=MT_prev_world[minMC].y+(MT_curr_camera[i].y-MT_prev_camera[minMC].y);
							MT_curr_world[i].z=MT_prev_world[minMC].z+(MT_curr_camera[i].x-MT_prev_camera[minMC].x)*sin(-aggre_RobotTH)+(MT_curr_camera[i].z-MT_prev_camera[minMC].z)*cos(-aggre_RobotTH);
						
						}
					}
					MT_curr_world.resize(DMT_curr_ok);//Depth取得可能数でリサイズ(マッチング中心世界座標)
				}//if(markerIds.size() <= 0)→END

				//新規テンプレート追加動作
				//同じ所に作成されたテンプレート削除する(画像座標を使って比較)
				//3回目のマッチング結果(Prev)と2回目に作成した新規テンプレート(curr)を比較する
				//新規テンプレートと旧テンプレートを比較して全ての旧テンプレートと距離がテンプレートサイズ以上離れていたら追加する
				//距離がテンプレートサイズ以内ならばPrev(旧テンプレート)を削除し新規テンプレートを追加する
				//複数重なる場合は最後
				for (int i = 0; i < DMT_curr2_ok; i++) 
				{
					double remave=0,minlengh=1000000;
					int minj;
					for (int j = 0; j < DMT_curr_ok; j++) 
					{
						length=sqrt((MT_curr2_pixel[i].x-MT_curr_pixel[j].x)*(MT_curr2_pixel[i].x-MT_curr_pixel[j].x)
									+(MT_curr2_pixel[i].y-MT_curr_pixel[j].y)*(MT_curr2_pixel[i].y-MT_curr_pixel[j].y));
						//std::cout <<"template_size="<<template_size<< std::endl;
						//std::cout <<"更新前:MT_curr_pixel["<<j<<"]="<<MT_curr_pixel[j]<< std::endl;
						//std::cout <<"更新前:MT_curr2_pixel["<<i<<"]="<<MT_curr2_pixel[i]<< std::endl;
						
						//std::cout<<"length"<<length<<std::endl;
						//更新動作(テンプレートのかぶりがあるとき→最も距離が近いテンプレートで更新する)
						if(length<template_size*5)
						{
							//std::cout <<"かぶり"<< std::endl;
							remave=1;
							if(minlengh>length)
							{
								minj=j;//最も距離が近いテンプレートの配列を保存
								minlengh=length;//最小距離の更新
							}
						}
					}//for (int j = 0; j < DMT_curr_ok; j++)→end
					//更新動作(テンプレートのかぶりが存在する時)
					if(remave==1)
					{
						//std::cout <<"更新動作"<<std::endl;
						MT_curr2_Templ[i] = MT_curr_Templ[minj];//Depth取得可能なマッチテンプレート
						MT_curr2_pixel[i] = MT_curr_pixel[minj];//Depth取得可能なマッチング中心画像座標
						MT_curr2_camera[i] = MT_curr_camera[minj];
						MT_curr2_world[i] = MT_curr_world[minj];
						//更新したCurrを削除する(重複防止)
						int j,k;
						for ( j = k = 0; j < DMT_curr_ok; j++) 
						{
							if(j!=minj)
							{
								MT_curr_Templ[k]=MT_curr_Templ[j];
								MT_curr_pixel[k]=MT_curr_pixel[j];
								MT_curr_camera[k]=MT_curr_camera[j];
								MT_curr_world[k++]=MT_curr_world[j];
							}
						}
						DMT_curr_ok=k;
					}
				}//for (int i = 0; i < DMT_curr2_ok; i++) →end
				//追加動作(残ったCurrをCurr2に追加する→残ったcurrには更新要素が無いためテンプレートがかぶってない)
				for (int j = 0; j < DMT_curr_ok; j++) 
				{
					MT_curr2_Templ[DMT_curr2_ok] = MT_curr_Templ[j];//Depth取得可能なマッチテンプレート
					MT_curr2_pixel[DMT_curr2_ok] = MT_curr_pixel[j];//Depth取得可能なマッチング中心画像座標
					MT_curr2_camera[DMT_curr2_ok] = MT_curr_camera[j];
					MT_curr2_world[DMT_curr2_ok] = MT_curr_world[j];
					DMT_curr2_ok=DMT_curr2_ok+1;//Depthが取得可能な全マッチテンプレート数
				}
				MT_curr2_pixel.resize(DMT_curr2_ok);//Depth取得可能数でリサイズ(マッチング中心画像座標)
				MT_curr2_camera.resize(DMT_curr2_ok);//Depth取得可能数でリサイズ(マッチング中心カメラ座標)
				MT_curr2_world.resize(DMT_curr2_ok);//Depth取得可能数でリサイズ(マッチング中心世界座標)
				//std::cout <<"全テンプレート数:DMT_prev_ok="<<DMT_curr2_ok<< std::endl;
				//temp<< DMT_curr2_ok<<"\n";

				for (int i = 0; i < DMT_curr2_ok; i++) 
				{
					//std::cout <<"MT_curr2_pixel["<<i<<"]="<<MT_curr2_pixel[i]<< std::endl;
					//テンプレートまでの距離と角度を求める(観測値)
					CameraLMT[i][0]=sqrt((MT_curr2_camera[i].x*MT_curr2_camera[i].x)+(MT_curr2_camera[i].z*MT_curr2_camera[i].z));
					CameraLMT[i][1]=atan2(MT_curr2_camera[i].x,MT_curr2_camera[i].z);
					//std::cout <<"CameraLMT[i][0]="<<CameraLMT[i][0]<< std::endl;
					//std::cout <<"CameraLMT[i][1]="<<CameraLMT[i][1]*180/M_PI<< std::endl;
				}
			}
		}
		//tf(観測特徴点)観測マーカー
		std::string target_maker_frame1 = "MT_curr2_world";//cameraとマーカー間のリンク
		geometry_msgs::Pose maker_pose1;

		//std::cout << "tf特徴点の世界座標:MT_curr2_world[i]={x="<< MT_curr2_world[0].x <<",y="<<MT_curr2_world[0].y<<",z="<<MT_curr2_world[0].z<<"}"<< std::endl;
		maker_pose1.position.x = MT_curr2_world[0].z;//Rvizと画像は座標系が異なるので注意
		maker_pose1.position.y = MT_curr2_world[0].x;
		maker_pose1.position.z = MT_curr2_world[0].y;
		maker_pose1.orientation.w = 1.0;

		static tf::TransformBroadcaster br_maker1;
		tf::Transform maker_transform1;
		poseMsgToTF(maker_pose1, maker_transform1);
		br_maker1.sendTransform(tf::StampedTransform(maker_transform1, ros::Time::now(), source_frame, target_maker_frame1));
	}

	//ロボット指令(山口先輩の内容)
	ros::NodeHandle nh;
	pub= nh.advertise<geometry_msgs::Twist>("/robot1/mobile_base/commands/velocity", 10);
	if(X_25==false&&TH_90==false&&Y_05==false)
	{
		robot_velocity.linear.x  = VX+2.2*(Des_RobotX-Act_RobotX)+2.2*(Des_RobotY-Act_RobotY);//実行指令値
		robot_velocity.angular.z=(Des_RobotTH-Act_RobotTH)*2.2;
		pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
		if(Act_RobotV<=0.001)
		{
			robot_velocity.linear.x=0;	//指令値の遅れを考慮(推定に使用する指令値)
		}
		else
		{
			robot_velocity.linear.x  = VX;
		} 
		//robot_velocity.linear.x  = VX;
		//std::cout <<"11"<< std::endl;

		robot_velocity.angular.z = 0.0;
		if(Act_RobotV<=0.001&&Act_RobotX>LX)	//停止命令
		{
			//std::cout <<"2"<< std::endl;

			X_25=true;
			robot_velocity.linear.x  = 0.0; // 並進速度vの初期化
			robot_velocity.angular.z = 0.0; // 回転速度ωの初期化
			pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
			usleep(7*100000);//0.5秒ストップ(マイクロ秒)
		}
	}
	else if(X_25==true&&TH_90==false&&Y_05==false)	//回転動作
	{
		robot_velocity.linear.x  =  0;
		robot_velocity.angular.z  =  THZ+(Des_RobotTH-Act_RobotTH)*2.2;//実行指令値
		pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
		robot_velocity.linear.x  =  0.0;
		robot_velocity.angular.z  =  THZ; 
		if(abs(encoder_odometry_.twist.twist.angular.z)<=0.05)
		{
			robot_velocity.angular.z  =  0;	//指令値の遅れを考慮(推定に使用する指令値)
		}
		else
		{
			robot_velocity.angular.z  =  THZ;
		} 
		//velocity.angular.z  =  THZ; 
		//std::cout <<"初期設定"<< std::endl;
		//std::cout <<"33"<< std::endl;

		if(Act_RobotTH>3.141592653/(2*omegaZ)&&abs(encoder_odometry_.twist.twist.angular.z)<=0.05)	//廊下
		{
			
			//std::cout <<"4"<< std::endl;
			TH_90=true;
			robot_velocity.linear.x  = 0.0; // 並進速度vの初期化
			robot_velocity.angular.z = 0.0; // 回転速度ωの初期化}//xが1以上になったら終了
			pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
			usleep(7*100000);//0.5秒ストップ(マイクロ秒)
		}
		//std::cout << "NoiseV=" <<velocity.angular.z<< std::endl;

	}
	else if(X_25==true&&TH_90==true&&Y_05==false)	//直進
	{
		robot_velocity.linear.x  = VX+2.2*(Des_RobotX-Act_RobotX)+2.2*(Des_RobotY-Act_RobotY);
		robot_velocity.angular.z=(Des_RobotTH-Act_RobotTH)*2.2;
		

		pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
		if(Act_RobotV<=0.001)
		{
			robot_velocity.linear.x=0;
			//std::cout <<"55"<< std::endl;
		}
		else
		{
			robot_velocity.linear.x  = VX;	// 並進速度vの初期化
		} 
		//robot_velocity.linear.x  = VX; // 並進速度vの初期化
		robot_velocity.angular.z = 0.0; // 回転速度ωの初期化}//xが1以上になったら終了
		//std::cout <<"5"<< std::endl;
		if(Act_RobotV<=0.001&&Act_RobotY>LY)
		{
			Y_05=true;
			robot_velocity.linear.x  = 0.0; // 並進速度vの初期化
			robot_velocity.angular.z = 0.0; // 回転速度ωの初期化}//xが1以上になったら終了
			pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
			usleep(7*100000);//0.5秒ストップ(マイクロ秒)
			//std::cout <<"6"<< std::endl;
		}
	}
	else if(X_25==true&&TH_90==true&&Y_05==true)
	{
		robot_velocity.linear.x  = 0.0; // 並進速度vの初期化
		robot_velocity.angular.z = 0.0; // 回転速度ωの初期化}//xが1以上になったら終了
		pub.publish(robot_velocity);    // 速度指令メッセージをパブリッシュ（送信）
	}
	Act_RobotV = encoder_odometry_.twist.twist.linear.x+encoder_odometry_.twist.twist.linear.y;//速度ベクトルの合成
	//std::cout << "Act_RobotV=" <<Act_RobotV<< std::endl;

	Des_RobotV=robot_velocity.linear.x+robot_velocity.linear.y;//速度ベクトルの合成
	//std::cout << "Des_RobotV=" <<Des_RobotV<< std::endl;
	std::vector<std::vector<double>> noise_param = {
		{0,5.46e-6},	//直進で生じる道のり
		{0,1.13e-6},	//回転で生じる道のり
		{0,1.72e-6},
		{0,3.15e-6}
		};
	static std::vector<std::vector<double>> noise_(noise_param.size());

	//この下にカルマンフィルタを実装する(できたら別関数にしたい)
	//カルマンフィルタ初期設定
	if(kaisu==0)
	{
		DES_Robot = cv::Mat_<double>::zeros(3, 1);//目標指令状態
		ACT_Robot = cv::Mat_<double>::zeros(3, 1);//雑音の影響を考慮した実際の状態
		
		Cov = (cv::Mat_<double>(3,3) << //共分散Q
			1e-10, 0,      0,
			0,     1e-10,  0,
			0,     0,      1e-10);
		I = cv::Mat_<double>::eye(3, 3);//単位行列
		//z = cv::Mat_<double>::zeros(2, 1);//センサーの観測値(仮)
		std::random_device seed;
		std::mt19937 engine(seed());
		std::vector<std::normal_distribution<>> dists(noise_param.size());
		for (size_t i = 0; i < noise_param.size(); i++)
		{
			std::normal_distribution<> dist(noise_param[i][0],noise_param[i][1]);
			dists[i] = dist;
		}
		
		for (size_t i = 0; i < noise_.size(); i++)
		{
			for (size_t j = 0; j < pnum; j++)
			{
				noise_[i].push_back(dists[i](engine));
			}
		}

		std::cout <<"初期設定"<< std::endl;
	}
	
	if(time0 != false)
	{
		for (int i = 0; i < pnum; i++)
		{
			//目標値(Des)にノイズを入れることで擬似的に実効値(Act)を作り出している

			// breez_[i]=robot_velocity.linear.x+noise_[0][i]*sqrt(abs(robot_velocity.linear.x)/realsec)+noise_[1][i]*sqrt(abs(robot_velocity.angular.z)/realsec);//ノイズ付き速度(山口先輩)
			// greed_[i]=robot_velocity.angular.z+noise_[2][i]*sqrt(abs(robot_velocity.linear.x)/realsec)+noise_[3][i]*sqrt(abs(robot_velocity.angular.z)/realsec);//ノイズ付き角速度（山口先輩）
			//std::cout << "greed_[i]" <<greed_[i]<< std::endl;
			
			breez_[i]=Act_RobotV+noise_[0][i]*sqrt(abs(Act_RobotV)/realsec)+noise_[1][i]*sqrt(abs(encoder_odometry_.twist.twist.angular.z)/realsec);//ノイズ付き速度(エンコーダ基準)（鈴木先輩）
			greed_[i]=encoder_odometry_.twist.twist.angular.z+noise_[2][i]*sqrt(abs(Act_RobotV)/realsec)+noise_[3][i]*sqrt(abs(encoder_odometry_.twist.twist.angular.z)/realsec);//ノイズ付き角速度（鈴木先輩）

			// ROS_INFO("%d, %f, %f", i, breez_[i], greed_[i]);
		}

		double breez__min = *min_element(begin(breez_), end(breez_));
		double breez__max = *max_element(begin(breez_), end(breez_));
		double greed__min = *min_element(begin(greed_), end(greed_));
		double greed__max = *max_element(begin(greed_), end(greed_));

		// double any_min = *min_element(begin(any), end(any));
		// double any_max = *max_element(begin(any), end(any));

		// ROS_INFO("linear, min:%f, max:%f", breez__min, breez__max);
		// ROS_INFO("angular, min:%f, max:%f", greed__min, greed__max);

		// ROS_INFO("noise");
		// for (int i = 0; i < noise_.size(); i++) 
		// {
		// 	double noise_min = *min_element(begin(noise_[i]), end(noise_[i]));
		// 	double noise_max = *max_element(begin(noise_[i]), end(noise_[i]));
		// 	ROS_INFO("	%d, min:%f, max:%f", i, noise_min, noise_max);
		// }

		//ロボットの状態方程式
		//雑音の影響を考慮した実際の状態(カルマン推定前)
		for(int v=0;v<pnum;v++)
		{
			double th = particles_[v].rotation.z;
			if(greed_[v]<1e-10)
			{
				particles_[v].position.x += (realsec*breez_[v]*cos(th));
				particles_[v].position.y += (realsec*breez_[v]*sin(th));
				particles_[v].rotation.z += (greed_[v]*realsec);
				
			}
			else
			{
				particles_[v].position.x += ((breez_[v]/greed_[v])*(sin(th+greed_[v]*realsec)-sin(th)));
				particles_[v].position.y += ((breez_[v]/greed_[v])*(-cos(th+greed_[v]*realsec)+cos(th)));
				particles_[v].rotation.z += (greed_[v]*realsec);
			}

			aggrex += particles_[v].position.x;
			aggrey += particles_[v].position.y;
			aggrez += particles_[v].rotation.z;
			
		}

		if(encoder_odometry_.twist.twist.angular.z==0)
		{
			Act_RobotX=Act_RobotX+(Act_RobotV*cos(Act_RobotTH)*realsec);
			Act_RobotY=Act_RobotY+(Act_RobotV*sin(Act_RobotTH)*realsec);
			Act_RobotTH=Act_RobotTH+(encoder_odometry_.twist.twist.angular.z*realsec);
		}
		else
		{
			Act_RobotX=Act_RobotX+((Act_RobotV/encoder_odometry_.twist.twist.angular.z)*(sin(Act_RobotTH+encoder_odometry_.twist.twist.angular.z*realsec)-sin(Act_RobotTH)));
			Act_RobotY=Act_RobotY+((Act_RobotV/encoder_odometry_.twist.twist.angular.z)*(-cos(Act_RobotTH+encoder_odometry_.twist.twist.angular.z*realsec)+cos(Act_RobotTH)));
			Act_RobotTH=Act_RobotTH+(encoder_odometry_.twist.twist.angular.z*realsec);
		}

		aggre_RobotX=aggrex/pnum;
		aggre_RobotY=aggrey/pnum;
		aggre_RobotTH=aggrez/pnum;
		aggrex=0;
		aggrey=0;
		aggrez=0;
		
		nav_msgs::Odometry estimate_odometry;
		estimate_odometry.header.frame_id = source_frame;
		estimate_odometry.header.stamp = ros::Time::now();
		estimate_odometry.pose.pose.position.x = aggre_RobotX;
		estimate_odometry.pose.pose.position.y = aggre_RobotY;
		estimate_odometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0,0,aggre_RobotTH);
		estimate_odometry_pub.publish(estimate_odometry);

		if (Act_RobotV<0.001)
		{
			// std::cout << "停止" << std::endl;
			// std::cout << "aggre_RobotX=" <<aggre_RobotX<< std::endl;
			// std::cout << "aggre_RobotY=" <<aggre_RobotY<< std::endl;
			// std::cout << "aggre_RobotTH=" <<aggre_RobotTH<< std::endl;
			// std::cout << "====================================" << std::endl;
			// printf("処理時間合計=%f\n", ALLrealsec);
		}
		
		if(Act_RobotX>0||Act_RobotY>0||Act_RobotTH>0)//ラグの調整
		{
			////目標指令状態
			if(robot_velocity.angular.z==0)
			{
				Des_RobotX=Des_RobotX+(Des_RobotV*cos(Des_RobotTH)*realsec);
				Des_RobotY=Des_RobotY+(Des_RobotV*sin(Des_RobotTH)*realsec);
				Des_RobotTH=Des_RobotTH+(robot_velocity.angular.z*realsec);
			}
			else
			{
				Des_RobotX=Des_RobotX+((Des_RobotV/robot_velocity.angular.z)*(sin(Des_RobotTH+robot_velocity.angular.z*realsec)-sin(Des_RobotTH)));
				Des_RobotY=Des_RobotY+((Des_RobotV/robot_velocity.angular.z)*(-cos(Des_RobotTH+robot_velocity.angular.z*realsec)+cos(Des_RobotTH)));
				Des_RobotTH=Des_RobotTH+(robot_velocity.angular.z*realsec);
			}
			//std::cout << "velocity.angular.z=" <<velocity.angular.z<< std::endl;
			//std::cout << "Des_RobotX=" <<Des_RobotX<< std::endl;
			//std::cout << "Des_RobotY=" <<Des_RobotY<< std::endl;
			//std::cout << "Des_RobotTH=" <<Des_RobotTH*180/M_PI<< std::endl;
		}
		DES_Robot= (cv::Mat_<double>(3,1) <<
											Des_RobotX,
											Des_RobotY,
											Des_RobotTH);
		std::vector<double> ss;//重みのリスト

		//更新ステップ(マーカー)
		for (int j = 0; j < markerIds.size(); j++)
		{
			double xxx;
			double yyy;
			double kankyori;
			double sanjikansuu;

			for (int i = 0; i < pnum; i++)
			{

				double x = particles_[i].position.x;
				double y = particles_[i].position.y;
				double th = particles_[i].rotation.z;
				double lu = sqrt((x-MarkerW[markerIds.at(j)](2,0))*(x-MarkerW[markerIds.at(j)](2,0))+(y-MarkerW[markerIds.at(j)](0,0))*(y-MarkerW[markerIds.at(j)](0,0))); //マーカーとパーティクルの距離
				double lu_x = x-MarkerW[markerIds.at(j)](2,0);//平均値のX成分（山口先輩）
				//luX[i]=MarkerW[markerIds.at(j)](2,0)-Est_RobotX[i];//平均値のX成分（鈴木先輩）      
				double lu_y = y-MarkerW[markerIds.at(j)](0,0);//平均値のY成分
				double lu_omega = atan2(lu_y,lu_x);
				if (lu_omega>0) lu_omega-=2*3.1415;

				kankyori=abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu));//(山口先輩)

				//yyy=0.000856;//角度分散（鈴木先輩）
				yyy=0.000464;//(山口先輩)

				xxx=lu*lu*0.000856;//距離分散
				
				//↓変曲点に着目した距離分散変動
				if (abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu))>0.5&&abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu))<1.0)
				{
					xxx=abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu))*abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu));
				}

				double kakurituX=1/(sqrt(2*3.1415*xxx))*exp(-((abs(CameraLM[markerIds.at(j)][0])-abs(lu))*(abs(CameraLM[markerIds.at(j)][0])-abs(lu)))/(2*xxx))+1e-100;//ランドマークまでの距離
				
				double kakurituY=1/(sqrt(2*3.1415*yyy))*exp(-((CameraLM[markerIds.at(j)][1]-((th+1.57)-abs(lu_omega)))*(CameraLM[markerIds.at(j)][1]-((th+1.57)-abs(lu_omega))))/(2*yyy))+1e-100;//（山口先輩）
				//double kakurituY=1/(sqrt(2*3.1415*yyy))*exp(-((CameraLM[markerIds.at(j)][1]-(-luOmega[i]-Est_RobotTH[i]))*(CameraLM[markerIds.at(j)][1]-(-luOmega[i]-Est_RobotTH[i])))/(2*yyy))+1e-100;//(鈴木先輩)
				if (lu_omega*th>0&&lu_omega>1.57)
				{
					kakurituY=1/(sqrt(2*3.1415*yyy))*exp(-((CameraLM[markerIds.at(j)][1]-(-lu_omega-(th-2*3.1415)))*(CameraLM[markerIds.at(j)][1]-(-lu_omega-(th-2*3.1415))))/(2*yyy))+1e-100;//
				}
				else if (lu_omega*th>0&&th<-1.57)
				{
					kakurituY=1/(sqrt(2*3.1415*yyy))*exp(-((CameraLM[markerIds.at(j)][1]-(-lu_omega-(th+2*3.1415)))*(CameraLM[markerIds.at(j)][1]-(-lu_omega-(th+2*3.1415))))/(2*yyy))+1e-100;//
				}
				else
				{
					//std::cout <<"発生せず" << std::endl;
				}
				
				if (CameraLM[markerIds.at(j)][0]==0)//0413変更
				{
					kakurituX=1;
				}
				if (CameraLM[markerIds.at(j)][1]==0)//0413変更
				{
					kakurituY=1;
				}
				//↓0415変曲点改良案
				if (abs(abs(CameraLM[markerIds.at(j)][0])-abs(lu))>1.0)
				{
					kakurituX=1;
				}
				
				particle_weight_[i]*=(kakurituX*kakurituY);
				if (particle_weight_[i]==0)
				{
					std::cout<<"尤度限界"<<std::endl;
				}
				//x成分とy成分の１変量ガウス分布を作り出し、それらを掛け合わせることで２変量正規分布とした
			}
			//更新ステップ(マーカー)終わり

			//(最大尤度と合計尤度)また最大尤度を取るパーティクル番号とその座標を表示させるプログラム(山口追記)※一応
			totalLikelihood = 0.0;
			double maxLikelihood = 0.0;
			maxLikelihoodParticleIdx = 0;

			for(int i = 0; i < pnum; i++)
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
				//std::cout << "weight[i]=" <<weight[i]<< std::endl;
				//std::cout << "totalLikelihood=" <<totalLikelihood<< std::endl;
			}
			averageLikelihood = totalLikelihood / pnum;//必要なかったかも
			//追加システム(最大尤度と合計尤度、重みの正規化と自己位置推定)終了
			double sssss=0;
			
		}
		//リサンプリング
		std::vector<double> sd;//重みの累積和
		double sum=0;

		for (int i = 0; i < pnum; i++)
		{
			if (particle_weight_[i] < 1e-100)
			{
				particle_weight_[i] = particle_weight_[i] + 1e-100;
			}
			
			ss.push_back(particle_weight_[i]);
		}
		
		for (int i = 0; i <pnum ; i++)
		{
			sum+=ss[i];
			sd.push_back(sum);
			//std::cout << "sd=" <<sd[i]<< std::endl;
		}//累積和　sd={1,3,6,10,15}
		//std::cout << "sum=" <<sum<< std::endl;

		double step=sd[pnum-1]/pnum;//(重みの合計)/(パーティクルの合計)

		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<double> distr(0,step);
		double r=distr(eng);//0~stepの間でランダムな値を抽出する関数(おそらく小数付き)
		//std::cout << "r=" <<r<< std::endl;

		int cur_pos=0;
		int math=0;
		std::vector<potbot_lib::Pose> ps;//新たに抽出するパーティクルのリスト

		while (ps.size() < pnum)//もとのパーティクル数と一致するまで
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
		
		for (int i = 0; i < pnum; i++)
		{
			particle_weight_.at(i)=1/pnum;
		}
		//リサンプリング（終わり）
	}//if(time0 != false)→end; //プログラム初回時カルマンは実行しない(Δtがないから)

	robot_velocity.linear.x  = 0.0; // 並進速度の初期化
	robot_velocity.angular.z = 0.0; // 回転速度の初期化

	// 画面表示
	// img_dst ARマーカー検出
	// img_dst2 構造線のみ
	// img_fld 特徴線のみ
	// img_FLD_TY 線の延長
	// img_tate 特徴点
	// img_line2 横縦の分類
	// img_line4 横縦の分類
	//cv::imshow(win_src, img_src);//(出力画像のタイトル,画像データ)
	cv::imwrite("src.png",img_src);
	cv::imshow(win_dstfld, img_dstfld);
	//cv::imshow(win_dst, img_dst);
	//cv::imshow(win_dst2, img_dst2);
	//cv::imwrite("dst2.png",img_dst2);
	//cv::imshow(win_fld, img_fld);
	//cv::imwrite("fld.png",img_fld);
	//cv::imshow(win_fld_ty,img_FLD_TY);
	//cv::imshow(win_tate, img_tate);
	//cv::imwrite("tate.png",img_tate);
	//cv::imshow(win_line2, img_line2);
	//cv::imwrite("line2.png",img_line2);
	//cv::imshow(win_line4, img_line4);

	//初回動作時＋検出時
	cv::swap(image_curr, image_prev);// image_curr を image_prev に移す（交換する）
	cv::swap(tate_point_curr, tate_point_prev);//縦線中点の画像座標を保存(tate_point_curr→tate_point_prev)
	cv::swap(TPCC, TPCP);//縦線中点の三次元カメラ座標を保存(TPCC→TPCP)
	cv::swap(TPCC_Templ, TPCP_Templ);//今のテンプレートを一つ前のテンプレートとして保存
	DTPP_ok=DTPC_ok;//テンプレート取得数を一つ前の取得数として保存()

	//2回目動作時
	if(Tracking==false)
	{
		DMT_prev_ok=DMT_curr_ok;//Depth取得可能マッチテンプレート数をキープ+(3回目以降はこれに一つ前のテンプレート要素を追加する)
		cv::swap(MT_curr_world, MT_prev_world);//マッチ中心カメラ座標
		cv::swap(MT_curr_camera,MT_prev_camera);//マッチ中心カメラ座標
		cv::swap(MT_curr_pixel, MT_prev_pixel);//マッチ中心画像座標
		cv::swap(MT_curr_Templ, MT_prev_Templ);//マッチ座標
		Tracking=true;//3回目動作
		std::cout<<"二回目"<<std::endl;
	}
	else if(Tracking == true)	//3回目以降動作時
	{
		//Depth取得可能マッチング要素キープ
		DMT_prev_ok=DMT_curr2_ok;
		cv::swap(MT_curr2_world,  MT_prev_world);//マッチ中心カメラ座標
		cv::swap(MT_curr2_camera, MT_prev_camera);//マッチ中心カメラ座標
		cv::swap(MT_curr2_pixel,  MT_prev_pixel);//マッチ中心画像座標
		cv::swap(MT_curr2_Templ,  MT_prev_Templ);//マッチ座標
	}
	geometry_msgs::PoseArray particles_msg;
	particles_msg.header.stamp = ros::Time::now();//追加
	particles_msg.header.frame_id = source_frame;//追加
	for(const auto& p:particles_) particles_msg.poses.push_back(potbot_lib::utility::get_pose(p));
	particles_pub.publish(particles_msg);

	kaisu++;
	time0=true;//一回目スキップ
	endTime=startTime;//動作終了時刻取得
	endTimeV1=startTimeV1;//動作終了時刻取得
	endTimeM1=startTimeM1;//動作終了時刻取得
	cv::waitKey(1);//ros::spinにジャンプする
}

void VMCLNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	encoder_odometry_ = *msg;
	encoder_odometry_.header.stamp = ros::Time::now();
	static tf2_ros::TransformBroadcaster dynamic_br;
	// geometry_msgs::PoseStamped origin;
	// origin.header.frame_id = source_frame;
	// origin.header.stamp = msg->header.stamp;
	// origin.pose = potbot_lib::utility::get_pose();
	// potbot_lib::utility::broadcast_frame(dynamic_br, msg->header.frame_id, origin);
	potbot_lib::utility::broadcast_frame(dynamic_br, source_frame, encoder_odometry_.header.frame_id, potbot_lib::utility::get_pose());

	pub_odometry_.publish(encoder_odometry_);
}

//メイン関数
int main(int argc,char **argv)
{
	ros::init(argc,argv,"marker2");//rosを初期化
	
	VMCLNode vmcl;

  	ros::spin();//トピック更新
			
	return 0;
}

#endif