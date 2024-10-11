#include <vmcl/particle.h>

namespace vmcl
{
	Particle::Particle(int particle_num)
	{
		initialize(particle_num);
	}

	void Particle::initialize(int particle_num)
	{
		std::random_device seed;
		std::mt19937 tmp(seed());
		eng_ = tmp;

		particles_.resize(particle_num);
		particle_weight_.resize(particle_num,1.0/double(particle_num));
		breez_.resize(particle_num);
		greed_.resize(particle_num);

		std::vector<std::vector<double>> v;
		setParticleNoiseParams(v);
	}

	void Particle::update(double linear_vel, double angular_vel)
	{	
		ros::Time timestamp_now = ros::Time::now();
		static ros::Time timestamp_pre = timestamp_now;
		double dt = timestamp_now.toSec() - timestamp_pre.toSec();
		timestamp_pre = timestamp_now;
		int particle_num = particles_.size();

		double v = linear_vel;
		double omega = angular_vel;

		if(dt>0)
		{
			for (int i = 0; i < particle_num; i++)
			{
				//目標値(Des)にノイズを入れることで擬似的に実効値(Act)を作り出している
				// breez_[i]=velocity_command_.linear.x+noise_[0][i]*sqrt(abs(velocity_command_.linear.x)/realsec)+noise_[1][i]*sqrt(abs(velocity_command_.angular.z)/realsec);//ノイズ付き速度(山口先輩)
				// greed_[i]=velocity_command_.angular.z+noise_[2][i]*sqrt(abs(velocity_command_.linear.x)/realsec)+noise_[3][i]*sqrt(abs(velocity_command_.angular.z)/realsec);//ノイズ付き角速度（山口先輩）
				
				breez_[i]=v+particle_noise_[0](eng_)*sqrt(abs(v)/dt)+particle_noise_[1](eng_)*sqrt(abs(omega)/dt);//ノイズ付き速度(エンコーダ基準)（鈴木先輩）
				greed_[i]=omega+particle_noise_[2](eng_)*sqrt(abs(v)/dt)+particle_noise_[3](eng_)*sqrt(abs(omega)/dt);//ノイズ付き角速度（鈴木先輩）

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

		for(int v=0;v<particle_num;v++)
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

	double Particle::weightFunc(double x,double xhat, double sigma)
	{
		double val = 1.0/sqrt(2*M_PI*pow(sigma,2))*exp(-pow(x-xhat,2)/(2*pow(sigma,2)));
		
		return val;
	}

	void Particle::weighting(const potbot_lib::Pose& agent_pose, const std::vector<potbot_lib::Pose>& marker_poses)
	{

		if (marker_poses.empty())
		{
			return;
		}

		// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(marker_poses[0]));

		Eigen::Affine3d marker_agent = marker_poses[0].to_affine();
		Eigen::Affine3d agent_world = agent_pose.to_affine();

		double distance_to_marker_from_agent = hypot(marker_poses[0].position.x, marker_poses[0].position.y);
		double angle_to_marker_from_agent = atan2(marker_poses[0].position.y, marker_poses[0].position.x);
		
		int particle_num = particles_.size();
		particle_weight_.resize(particle_num);
		for (size_t i = 0; i < particle_num; i++)
		{
			Eigen::Affine3d particle_world = particles_[i].to_affine();
			Eigen::Affine3d particle_agent = particle_world * agent_world.inverse();
			Eigen::Affine3d marker_particle = marker_agent * particle_agent.inverse();
			potbot_lib::Pose marker_particle_pose(marker_particle);

			double distance_to_marker_from_particle = hypot(marker_particle_pose.position.x, marker_particle_pose.position.y);
			double angle_to_marker_from_particle = atan2(marker_particle_pose.position.y, marker_particle_pose.position.x);
			
			double variance_distance = variance_distance_;
			double variance_angle = variance_angle_;
			
			double wd = weightFunc(distance_to_marker_from_agent, distance_to_marker_from_particle, variance_distance);
			double wth = weightFunc(angle_to_marker_from_agent, angle_to_marker_from_particle, variance_angle);
			// ROS_INFO_STREAM(wd<<"  "<<wth);
			double weight = wd*wth;
			

			particle_weight_[i] = weight;
		}

		double sum = 0;
		for (const auto& w:particle_weight_) sum+=w;
		for (auto& w:particle_weight_) w/=sum;
		for (const auto& w:particle_weight_) sum+=w;
		// ROS_INFO("%f, %f, %f",sum,distance_to_marker_from_agent,angle_to_marker_from_agent);
		
	}

	void Particle::resampling()
	{
		int particle_num = particles_.size();
		double xsum=0,ysum=0,thsum=0;
		for (const auto& p:particles_)
		{
			xsum+=p.position.x;
			ysum+=p.position.y;
			thsum+=p.rotation.z;
		}
		
		double xhat = xsum/particle_num;
		double yhat = ysum/particle_num;
		double thhat = thsum/particle_num;
		
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

		for (int i = 0; i < particle_num; i++)
		{
			if (particle_weight_[i] < 1e-100)
			{
				particle_weight_[i] = particle_weight_[i] + 1e-100;
			}
			
			ss.push_back(particle_weight_[i]);
		}
		
		for (int i = 0; i <particle_num ; i++)
		{
			sum+=ss[i];
			sd.push_back(sum);
			//std::cout << "sd=" <<sd[i]<< std::endl;
		}//累積和　sd={1,3,6,10,15}
		//std::cout << "sum=" <<sum<< std::endl;

		double step=sd[particle_num-1]/particle_num;//(重みの合計)/(パーティクルの合計)

		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<double> distr(0,step);
		double r=distr(eng);//0~stepの間でランダムな値を抽出する関数(おそらく小数付き)
		//std::cout << "r=" <<r<< std::endl;

		int cur_pos=0;
		int math=0;
		std::vector<potbot_lib::Pose> ps;//新たに抽出するパーティクルのリスト

		while (ps.size() < particle_num)//もとのパーティクル数と一致するまで
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
		
		for (int i = 0; i < particle_num; i++)
		{
			particle_weight_.at(i)=1.0/(double)particle_num;
		}
		//リサンプリング（終わり）
	}

	void Particle::setParticleNoiseParams(const std::vector<std::vector<double>>& noise_params)
	{
		bool is_param_error = false;
		if (noise_params.size() != 4)
		{
			is_param_error = true;
		}
		else
		{
			for (const auto& p:noise_params)
			{
				if (p.size() != 2)
				{
					is_param_error = true;
					break;
				}
			}
		}

		if (is_param_error)
		{
			particle_noise_params_ = {{0,0.01},{0,0.01},{0,0.01},{0,0.01}};
		}
		else
		{
			particle_noise_params_ = noise_params;
		}
		
		particle_noise_.resize(particle_noise_params_.size());
		for (size_t i = 0; i < particle_noise_params_.size(); i++)
		{
			std::normal_distribution<> dist(particle_noise_params_[i][0],particle_noise_params_[i][1]);
			particle_noise_[i] = dist;
		}
	}

	std::vector<potbot_lib::Pose> Particle::getParticles()
	{
		return particles_;
	}

	potbot_lib::Pose Particle::getEstimatedPose()
	{
		potbot_lib::Pose estimated_pose;
		for (size_t i = 0; i < particles_.size(); i++)
		{
			estimated_pose = estimated_pose + particles_[i]*particle_weight_[i];
		}
		// potbot_lib::utility::print_pose(potbot_lib::utility::get_pose(estimated_pose));
		return estimated_pose;
	}
}