//
// Created by cx on 25-8-11.
//
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <grasp_generator/ObjectDetection2D.h>
#include <grasp_generator/GraspPose.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

using namespace Eigen;

class GraspGenerator
{
public:
    GraspGenerator() : nh_("~"), it_(nh_)
    {
        // 参数
        nh_.param("gripper_min", gripper_min_, 0.02f);
        nh_.param("gripper_max", gripper_max_, 0.08f);
        nh_.param("safety_margin", safety_margin_, 0.01f);
        nh_.param("grasp_offset", grasp_offset_, 0.005f);
        nh_.param("min_points", min_points_, 10);
        nh_.param("use_shallowest_point", use_shallowest_point_, true);

        // 订阅
        cam_info_sub_ = nh_.subscribe("/camera/depth/camera_info", 1,
                                     &GraspGenerator::camInfoCallback, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
                                  &GraspGenerator::depthCallback, this);
        det_sub_ = nh_.subscribe("/object_detections", 10,
                                &GraspGenerator::detectionCallback, this);

        // 发布
        grasp_pub_ = nh_.advertise<grasp_generator::GraspPose>("/grasp_poses", 10);

        ROS_INFO("Grasp Generator Node Initialized (No Mask Version)");
    }

    void camInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        if (!cam_info_received_)
        {
            // 提取相机内参
            K_ << msg->K[0], msg->K[1], msg->K[2],
                  msg->K[3], msg->K[4], msg->K[5],
                  msg->K[6], msg->K[7], msg->K[8];
            K_inv_ = K_.inverse();
            cam_info_received_ = true;
            ROS_INFO("Camera parameters received");
        }
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            cv_depth_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void detectionCallback(const grasp_generator::ObjectDetection2DConstPtr& msg)
    {
        if (!cam_info_received_ || !cv_depth_ptr_)
        {
            ROS_WARN("Waiting for camera info or depth image");
            return;
        }

        try
        {
            // 提取检测信息
            std::vector<int> bbox = msg->bbox;

            // 生成抓取位姿
            Matrix4f grasp_pose;
            float width, score;
            if (generateGrasp(bbox, cv_depth_ptr_->image, grasp_pose, width, score))
            {
                // 创建抓取消息
                grasp_generator::GraspPose grasp_msg;
                grasp_msg.header.stamp = ros::Time::now();
                grasp_msg.header.frame_id = "camera_depth_optical_frame";

                // 位置
                Vector3f position = grasp_pose.block<3,1>(0,3);
                grasp_msg.pose.position.x = position(0);
                grasp_msg.pose.position.y = position(1);
                grasp_msg.pose.position.z = position(2);

                // 方向 (旋转矩阵转四元数)
                Matrix3f R = grasp_pose.block<3,3>(0,0);
                Quaternionf q(R);
                grasp_msg.pose.orientation.x = q.x();
                grasp_msg.pose.orientation.y = q.y();
                grasp_msg.pose.orientation.z = q.z();
                grasp_msg.pose.orientation.w = q.w();

                grasp_msg.width = width;
                grasp_msg.score = score;
                grasp_msg.shape = msg->shape;
                grasp_msg.color = msg->color;

                grasp_pub_.publish(grasp_msg);
                ROS_INFO("Published grasp pose for %s %s", msg->color.c_str(), msg->shape.c_str());
            }
        }
        catch (const std::exception& e)
        {
            ROS_ERROR("Grasp generation error: %s", e.what());
        }
    }

    Vector3f pixelToCamera(float u, float v, float depth)
    {
        Vector3f uv_hom(u, v, 1.0f);
        Vector3f xyz = depth * (K_inv_ * uv_hom);
        return xyz;
    }

    bool generateGrasp(const std::vector<int>& bbox, const cv::Mat& depth_map,
                      Matrix4f& grasp_pose, float& grip_width, float& stability_score)
    {
        int x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
        int roi_width = x2 - x1;
        int roi_height = y2 - y1;

        // 提取检测框内的有效点
        std::vector<Vector3f> points;
        std::vector<float> depths;
        Vector3f shallowest_point(0,0,0);
        float min_depth = std::numeric_limits<float>::max();

        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                float d = depth_map.at<float>(y, x);
                if (d > 0)
                {
                    Vector3f p = pixelToCamera(x, y, d);
                    points.push_back(p);
                    depths.push_back(d);

                    // 记录最浅点
                    if (d < min_depth)
                    {
                        min_depth = d;
                        shallowest_point = p;
                    }
                }
            }
        }

        if (points.size() < min_points_)
        {
            ROS_WARN("Not enough valid points: %zu < %d", points.size(), min_points_);
            return false;
        }

        // 计算中值深度
        std::vector<float> sorted_depths = depths;
        std::sort(sorted_depths.begin(), sorted_depths.end());
        float median_depth = sorted_depths[sorted_depths.size() / 2];

        // 确定抓取点
        Vector3f grasp_point;
        if (use_shallowest_point_)
        {
            grasp_point = shallowest_point;
        }
        else
        {
            // 使用中心点
            int center_u = (x1 + x2) / 2;
            int center_v = (y1 + y2) / 2;
            grasp_point = pixelToCamera(center_u, center_v, median_depth);
        }

        // 使用PCA估计法线
        // 计算质心
        Vector3f centroid(0, 0, 0);
        for (const auto& p : points) centroid += p;
        centroid /= points.size();

        // 计算协方差矩阵
        Matrix3f cov = Matrix3f::Zero();
        for (const auto& p : points)
        {
            Vector3f diff = p - centroid;
            cov += diff * diff.transpose();
        }
        cov /= points.size();

        // 特征分解
        SelfAdjointEigenSolver<Matrix3f> eig(cov);
        Vector3f normal = eig.eigenvectors().col(0); // 最小特征值对应的特征向量

        // 确保法线朝向相机
        if (normal(2) > 0) normal = -normal;
        normal.normalize();

        // 计算夹爪宽度
        float focal_x = K_(0, 0);
        float physical_width = (roi_width * median_depth) / focal_x;
        grip_width = std::max(gripper_min_, std::min(gripper_max_, 1.2f * physical_width));

        // 构建抓取位姿
        // Z轴: 法线方向
        Vector3f z_axis = normal;

        // X轴: 与物体宽度方向对齐
        Vector3f cam_x(1, 0, 0);
        Vector3f x_axis = cam_x - cam_x.dot(z_axis) * z_axis;
        if (x_axis.norm() < 1e-6)
        {
            x_axis = Vector3f(0, 1, 0); // 备用方向
        }
        x_axis.normalize();

        // Y轴: 与X轴和Z轴垂直
        Vector3f y_axis = z_axis.cross(x_axis);
        y_axis.normalize();

        // 构建旋转矩阵
        Matrix3f R;
        R.col(0) = x_axis;
        R.col(1) = y_axis;
        R.col(2) = z_axis;

        // 构建平移矩阵 (稍微抬升)
        Vector3f translation = grasp_point + grasp_offset_ * z_axis;

        // 完整位姿
        grasp_pose = Matrix4f::Identity();
        grasp_pose.block<3,3>(0,0) = R;
        grasp_pose.block<3,1>(0,3) = translation;

        // 计算稳定性分数
        float depth_variance = 0.0f;
        float sum = 0.0f, sum_sq = 0.0f;
        for (float d : depths)
        {
            sum += d;
            sum_sq += d * d;
        }
        depth_variance = (sum_sq - sum * sum / depths.size()) / depths.size();

        float coverage = depths.size() / static_cast<float>(roi_width * roi_height);
        stability_score = exp(-10 * depth_variance) * coverage;

        return true;
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber cam_info_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Subscriber det_sub_;
    ros::Publisher grasp_pub_;

    cv_bridge::CvImagePtr cv_depth_ptr_;
    Matrix3f K_;
    Matrix3f K_inv_;
    bool cam_info_received_ = false;

    float gripper_min_;
    float gripper_max_;
    float safety_margin_;
    float grasp_offset_;
    int min_points_;
    bool use_shallowest_point_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "grasp_generator_node");
    GraspGenerator generator;
    ros::spin();
    return 0;
}