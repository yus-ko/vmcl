#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header  # headerのあるメッセージタイプを使う場合
from sensor_msgs.msg import Imu  # 例としてIMUメッセージを使いますが、必要なメッセージタイプに変更
from nav_msgs.msg import Odometry

# コールバック関数
def callback(msg):
    # 受け取ったメッセージのheader.stampを現在時刻に更新
    if hasattr(msg, 'header'):
        msg.header.stamp = rospy.Time.now()

    # パブリッシュ
    pub.publish(msg)

if __name__ == '__main__':
    try:
        # ノードの初期化
        rospy.init_node('timestamp_modifier', anonymous=True)

        # サブスクライブするトピック名とメッセージ型を設定
        input_topic = "/robot1/odom"  # 例：入力トピック
        output_topic = "/robot1/odom/sync"  # 例：出力トピック

        # 再パブリッシュするためのパブリッシャー
        pub = rospy.Publisher(output_topic, Odometry, queue_size=10)  # 例ではImuメッセージを使用

        # サブスクライブして、コールバックでheader.stampを更新
        rospy.Subscriber(input_topic, Odometry, callback)

        # ノードが終了するまで待機
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
