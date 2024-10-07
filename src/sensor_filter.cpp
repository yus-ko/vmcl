#include <vmcl/vmcl.h>

ros::Publisher g_pub_noise_twist;
ros::Publisher g_pub_time_sync_odometry;
double g_variance = 0.1;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	nav_msgs::Odometry odom_msg = *msg;
	geometry_msgs::Twist vel = odom_msg.twist.twist;

	static std::random_device seed;
	static std::mt19937 engine(seed());
	static std::normal_distribution<> dist(0.0, g_variance);
	vel.linear.x += dist(engine);
	vel.angular.z += dist(engine);

	g_pub_noise_twist.publish(vel);

	odom_msg.header.stamp = ros::Time::now();
	g_pub_time_sync_odometry.publish(odom_msg);
}

int main(int argc,char **argv)
{
	ros::init(argc,argv,"add_noise");


	ros::NodeHandle pnh("~");
	pnh.getParam("variance", g_variance);
	
	ros::NodeHandle nh;
	ros::Subscriber sub_odom = nh.subscribe("odom", 1, odomCallback);
	g_pub_noise_twist = nh.advertise<geometry_msgs::Twist>("noise_twist",1000);
	g_pub_time_sync_odometry = nh.advertise<nav_msgs::Odometry>("odom/time_sync",1000);

	ros::spin();
			
	return 0;
}