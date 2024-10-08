#ifndef _VMCL_PARTICLE_H
#define _VMCL_PARTICLE_H

#include <potbot_lib/utility_ros.h>

namespace vmcl
{
	class Particle
	{
		private:
			potbot_lib::Pose estimated_pose_;	//推定位置
			std::vector<potbot_lib::Pose> particles_;	//パーティクルの位置
			std::vector<double> particle_weight_;//各パーティクルに対する重み
			std::vector<double> breez_;//ノイズ付きパーティクル速度
			std::vector<double> greed_;//ノイズ付きパーティクル
			std::vector<std::vector<double>> noise_;
			std::vector<std::vector<double>> noise_params_;
		public:
			Particle(int particle_num = 100);
			~Particle(){};

			void initialize(int particle_num = 100);
			void update(double linear_vel, double angular_vel);
			void weighting(){};
			void estimatePose(){};
			void resampling();

			void setNoiseParams(std::vector<std::vector<double>> noise_params);

			std::vector<potbot_lib::Pose> getParticles(){};
			potbot_lib::Pose getEstimatedPose(){};

	};
}

#endif