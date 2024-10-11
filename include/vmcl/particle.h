#ifndef _VMCL_PARTICLE_H
#define _VMCL_PARTICLE_H

#include <potbot_lib/utility_ros.h>

namespace vmcl
{
	class Particle
	{
		private:
			std::vector<potbot_lib::Pose> particles_;	//パーティクルの位置
			std::vector<double> particle_weight_;//各パーティクルに対する重み
			std::vector<double> breez_;//ノイズ付きパーティクル速度
			std::vector<double> greed_;//ノイズ付きパーティクル
			std::mt19937 eng_;
			std::vector<std::normal_distribution<>> particle_noise_;
			std::vector<std::vector<double>> particle_noise_params_;

			double variance_distance_ = 0.1;
			double variance_angle_ = 0.1;

			double weightFunc(double x, double xhat, double sigma);
		public:
			Particle(int particle_num = 100);
			~Particle(){};

			void initialize(int particle_num = 100);
			void update(double linear_vel, double angular_vel);
			void weighting(const potbot_lib::Pose& agent_pose, const std::vector<potbot_lib::Pose>& marker_poses);
			void resampling();

			void setParticleNoiseParams(const std::vector<std::vector<double>>& noise_params);
			void setVariance(double distance, double angle){
				variance_distance_ = distance;
				variance_angle_ = angle;
			};

			std::vector<potbot_lib::Pose> getParticles();
			potbot_lib::Pose getEstimatedPose();

	};
}

#endif