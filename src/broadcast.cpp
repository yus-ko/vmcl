#include <vmcl/vmcl.h>

int main(int argc,char **argv)
{
	ros::init(argc,argv,"marker_broadcaster");

	std::vector<vmcl::Marker> markers_truth; //マーカーの世界座標

	ros::NodeHandle pnh("~");
	XmlRpc::XmlRpcValue markers;
	pnh.getParam("markers", markers);
	for (size_t i = 0; i < markers.size(); i++)
	{
		vmcl::Marker marker;
		marker.id = static_cast<int>(markers[i]["id"]);
		marker.frame_id = static_cast<std::string>(markers[i]["frame_id"]);
		marker.pose.position.x = static_cast<double>(markers[i]["pose"]["x"]);
		marker.pose.position.y = static_cast<double>(markers[i]["pose"]["y"]);
		marker.pose.position.z = static_cast<double>(markers[i]["pose"]["z"]);
		marker.pose.rotation.x = static_cast<double>(markers[i]["pose"]["roll"]);
		marker.pose.rotation.y = static_cast<double>(markers[i]["pose"]["pitch"]);
		marker.pose.rotation.z = static_cast<double>(markers[i]["pose"]["yaw"]);

		markers_truth.push_back(marker);
	}

	ros::Rate rate(60);

	tf2_ros::TransformBroadcaster dynamic_br;
	while (ros::ok())
	{
		for (const auto& m:markers_truth)
		{
			std::string child_frame_id = "marker_" + std::to_string(m.id);
			geometry_msgs::Pose pose = potbot_lib::utility::get_pose(m.pose);
			potbot_lib::utility::broadcast_frame(dynamic_br, m.frame_id, child_frame_id, pose);
		}
		ros::spinOnce();
		rate.sleep();
	}
			
	return 0;
}