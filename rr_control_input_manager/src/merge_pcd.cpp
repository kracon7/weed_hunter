/*
* Run this code
* This code will print the data sets in screen if do not sepecify >> filename
* ./bin/velodyne_data >> output_file
*/

#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdint.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>


using namespace std;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

PointCloud::Ptr globalCloud(new PointCloud);

pcl::visualization::CloudViewer viewer("Sparse Point Clouds");
bool status = true;
void *pcl_viewer(void *arg)
{
    while(true)
    {
        if(globalCloud->points.size()>0 && status)
        {
            viewer.showCloud(globalCloud);
            usleep(100);
        }
    }
}



class PointCloudMerger
{

    private:
    std::string target_frame_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tfListener_;
    ros::Subscriber sub;
    int counter;
    int every_frames;

    public:
    PointCloudMerger(ros::NodeHandle *nh):
    tfListener_(tf_buffer_),
    target_frame_("map")
    {
        sub = nh->subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 70,
                        &PointCloudMerger::pointCloudCallback, this);
        counter = 0;
        every_frames = 20;
    }

    void pointCloudCallback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input)
    {
        // Process every certain frames
        if(counter%every_frames == 0)
        {
            geometry_msgs::TransformStamped transform;
            sensor_msgs::PointCloud2 cloud_world;
            try
            {
                transform = tf_buffer_.lookupTransform(target_frame_, input->header.frame_id,
                                        input->header.stamp, ros::Duration(1));
                tf2::doTransform(*input, cloud_world, transform);
            }
            catch (tf2::TransformException& ex)
            {
                ROS_WARN("%s", ex.what());
                return;
            }

            // Container for original & filtered data
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(cloud_world, pcl_pc2);
            PointCloud::Ptr pcl_cloud(new PointCloud);
            pcl::fromPCLPointCloud2(pcl_pc2, *pcl_cloud);

            *globalCloud += *pcl_cloud;

            ROS_INFO("Frame %d processed.", counter);

            sleep(0.1);    
        }

        counter++;
        
    }

};


int main(int argc, char ** argv)
{
    //create pcl viewer thread
    pthread_t pcl_thread;
    pthread_create(&pcl_thread, NULL, pcl_viewer, (void *)NULL);

    ros::init(argc, argv, "pcd_listener");
    ros::NodeHandle nh;
    PointCloudMerger pm(&nh); //Construct class
    ros::spin(); // Run until interupted 
    return 0;
};

