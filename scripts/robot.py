#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm, uniform
import copy


'''----- 雑音などを考慮したロボットクラス -----'''
##++++++++++++++++++++++++++##
## (1) 雑音                 ##
## (2) バイアス             ##
## (3) スタック(引っ掛かり) ##
## (4) 誘拐                 ##
##++++++++++++++++++++++++++##
# pose                 : 姿勢 np.array([x, y, theta])
# agent                : エージェント
# sensor               : センサ
# color                : 描画するときの色
# noise_per_stds       : 1[m]あたりの小石の数
# noise_std            : 小石を踏んだときにロボットの向き(theta[deg])に発生する雑音の標準偏差
# bias_rate_stds       : バイアスの係数をドローするためのガウス分布の標準偏差 (速度, 角速度)
# expected_stuck_time  : スタックまでの時間の期待値
# expected_escape_time : スタックから脱出するまでの時間の期待値
# expected_kidnap_time : 誘拐が起こるまでの時間の期待値
# kidnap_range_x       : 誘拐後にロボットが置かれる位置の範囲(x)
# kidnap_range_y       : 誘拐後にロボットが置かれる位置の範囲(y)
class Robot(IdealRobot):
    def __init__(self, pose, agent=None, sensor=None, color="black",
                 noise_per_meter=5, noise_std=math.pi/60.0,
                 bias_rate_stds=(0.1,0.1),
                 expected_stuck_time=1e100, expected_escape_time=1e-100,
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)):
        super().__init__(pose, agent, sensor, color)

        ##(1) 雑音
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter)) #指数分布のオブジェクト生成
                                                                     #1e-100は noise_per_meterが0でもいいように.
        self.distance_until_noise = self.noise_pdf.rvs()             #最初に小石を踏むまでの道のりをセット
        self.theta_noise = norm(scale=noise_std) #ガウス分布のオブジェクト生成(ロボットの向きに発生する雑音に使用)

        ##(2) バイアス
        # ロボット固有のバイアスを決定
        self.bias_rate_nu    = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        ##(3) スタック 
        self.stuck_pdf  = expon(scale=expected_stuck_time)  #指数分布
        self.escape_pdf = expon(scale=expected_escape_time) #指数分布
        self.time_until_stuck  = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False

        ##(4) 誘拐
        self.kidnap_pdf = expon(scale=expected_kidnap_time) #指数分布
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0],ry[0],0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi)) #一様分布
        #loc: 下限, scale: 上限
        

    ### 条件を満たしたらノイズを加える #################################################################
    # pose          : 姿勢 np.array([x, y, theta])
    # nu            : 速度
    # omega         : 角速度
    # time_interval : 離散時間1ステップの時間
    #--------------------------------------------------
    # pose          : 姿勢 
    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:                   #小石を踏んだら
            self.distance_until_noise += self.noise_pdf.rvs()  #次に小石を踏むまでの道のりを求める 
            pose[2] += self.theta_noise.rvs()                  #ロボットの向きに発生する雑音を求め, 加える

        return pose


    ### バイアスを掛けた値を計算 ######################################################################
    # nu    : 速度
    # omega : 角速度
    #------------------------------------------------------------
    # RET   : バイアスを掛けた速度, バイアスを掛けた角速度
    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega


    ### スタックしていたら姿勢の変化を止める #########################################################
    # nu            : 速度
    # omega         : 角速度
    # time_interval : 離散時間1ステップの時間
    #------------------------------------------------------------------------
    # RET           : 姿勢 (スタックしていたら速度, 角速度ともにゼロを返す)
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck) #Trueは1, Falseは0として扱われる


    ### 誘拐が起きたら誘拐後の位置を返す #############################################################
    # pose          : 姿勢
    # time_interval : 離散時間1ステップの時間 
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:                    #誘拐が起きたら
            self.time_until_kidnap += self.kidnap_pdf.rvs()  #次に誘拐されるまでの時間を決めて
            return np.array(self.kidnap_dist.rvs()).T        #誘拐後の位置を返す
        else:
            return pose


    ### 離散時間を1ステップ進める #####################################################################
    # time_interval : 離散時間1ステップの時間
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)





'''----- 観測に対する不確かさを考慮したカメラクラス -----'''
##++++++++++++++++++++##
## (1) 雑音           ##
## (2) バイアス       ##
## (3) ファントム     ##
## (4) 見落とし       ##
## (5) オクルージョン ##
##++++++++++++++++++++##
# env_map                   : 地図のオブジェクト
# distance_range            : 計測可能距離
# direction_range           : 計測可能角度
# distance_noise_rate       : 距離に加える雑音の標準偏差
# direction_noise           : 方角に加える雑音の標準偏差
# distance_bias_rate_stddev : 距離に関するバイアスの量を決定するときのガウス分布の標準偏差
# direction_bias_stddev     : 方角に関するバイアスの量を決定するときのガウス分布の標準偏差
# phantom_prob              : 各ランドマークのファントムが出現する確率
# phantom_range_x           : ファントムが出現する位置の範囲(x)
# phantom_range_y           : ファントムが出現する位置の範囲(y)
# oversight_prob            : 見落とす確率
# occlusion_prob            : オクルージョンが起こる確率
class Camera(IdealCamera):
    def __init__(self, env_map,
                 distance_range=(0.5,6.0), direction_range=(-math.pi/3,math.pi/3),
                 distance_noise_rate=0.1, direction_noise=math.pi/90,
                 distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi/90,
                 phantom_prob=0.0, phantom_range_x=(-5.0,5.0), phantom_range_y=(-5.0,5.0),
                 oversight_prob=0.0,
                 occlusion_prob=0.0):
        super().__init__(env_map, distance_range, direction_range)

        ##(1) 雑音
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise     = direction_noise

        ##(2) バイアス
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias         = norm.rvs(scale=direction_bias_stddev)

        ##(3) ファントム
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1]-rx[0], ry[1]-ry[0])) #一様分布
        self.phantom_prob = phantom_prob

        ##(4) 見落とし
        self.oversight_prob = oversight_prob

        ##(5) オクルージョン
        self.occlusion_prob = occlusion_prob


    ### 雑音を混入したセンサ値を返す ###################################################################
    # relpos : センサ値 (極座標????????????)
    #-----------------------------------
    # RET    : 雑音を混入したセンサ値
    def noise(self, relpos):
        ## loc: 期待値, scale: 標準偏差
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T


    ### バイアスをセンサ値に加える ####################################################################
    # relpos : センサ値
    #------------------------------------
    # RET    : バイアスを加えたセンサ値
    def bias(self, relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,
                                  self.direction_bias]).T


    ### ファントムを観測 #############################################################################
    # cam_pose : カメラの姿勢
    # relpos   : 物体の位置
    #---------------------------------------------
    # RET      : ファントム, または観測物体の位置
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            #return self.relative_polar_pos(cam_pose, pos)
            return IdealCamera.observation_function(cam_pose, pos)
        else:
            return relpos

        
    ### 見落とし #####################################################################################
    # relpos   : 物体の位置
    #-------------------------------------------------
    # RET      : None(見落とし), または観測物体の位置
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos


    ### オクルージョン(ここでは真の値よりも大きくなってしまう現象を扱う) #############################
    # relpos   : 物体の位置
    #-------------------------------------------------------------------
    # RET      : オクルージョンが起きたときの位置, または観測物体の位置
    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ## 雑音は最終的なセンサ値が, 現在の値と計測可能な最大距離の間の値になるように一様分布で決定する
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1]-relpos[0])
            return np.array([ell, relpos[1]]).T
        else:
            return relpos


    ### データ #######################################################################################
    # cam_pose : カメラの姿勢
    #-------------------------------------------
    # RET      : observed : 観測物体のリスト
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            #z = self.relative_polar_pos(cam_pose, lm.pos)
            z = IdealCamera.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed



    
    
def main():

    #####{ 移動に対する不確かさの要因の実装
    # world = World(30, 0.1)
    
    ### ランダムな雑音
    # for i in range(100):
    #     circling = Agent(0.2, 10.0/180*math.pi)
    #     r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray")
    #     world.append(r)
    
    ### バイアス
    # circling = Agent(0.2, 10.0/180*math.pi)
    # nobias_robot = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray")
    # world.append(nobias_robot)
    # biased_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling,
    #                      color="red", noise_per_meter=0, bias_rate_stds=(0.2,0.2))
    # world.append(biased_robot)
    
    ### スタック
    # circling = Agent(0.2, 10.0/180*math.pi)
    # for i in range(100):
    #     r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray",
    #               noise_per_meter=0, bias_rate_stds=(0.0,0.0),
    #               expected_stuck_time=60.0, expected_escape_time=60.0)
    #     world.append(r)
    
    # r = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red")
    # world.append(r)
    
    ### 誘拐
    # circling = Agent(0.2, 10.0/180*math.pi)
    # for i in range(100):
    #     r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray",
    #               noise_per_meter=0, bias_rate_stds=(0.0,0.0), expected_kidnap_time=5)
    #     world.append(r)
    
    # r = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red")
    # world.append(r)
    
    # world.draw()
    #####} 移動に対する不確かさの要因の実装
    
    
    #####{ 観測に対する不確かさの要因の実装
    world = World(30, 0.1)
    
    ### センサ値に対する雑音
    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)
    
    circling = Agent(0.2, 10.0/180*math.pi)
    r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling)
    world.append(r)
    
    world.draw()
    #####} 観測に対する不確かさの要因の実装




    
if __name__ == '__main__':
    main()
    
