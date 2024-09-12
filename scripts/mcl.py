#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from robot import *
from scipy.stats import multivariate_normal
import random
import copy



'''----- パーティクルクラス -----'''
# init_pose : 初期姿勢  np.array([x, y, theta])
# weight    : 重み
class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight


    ### パーティクルを動かす ############################################################################
    # nu             : 速度
    # omega          : 角速度
    # time           : 離散時間1ステップの時間
    # noise_rate_pdf : ロボットの動きにより生じる雑音のpdf
    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns           = noise_rate_pdf.rvs() #雑音をドロー
        noised_nu    =    nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose    = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose) #姿勢を更新


    ### 観測したセンサ値を使って尤度計算 ###############################################################
    # observation       : センサ値
    # envmap            : 地図のオブジェクト
    # distance_dev_rate : 距離に加える雑音の標準偏差
    # direction_dev     : 方角に加える雑音の標準偏差
    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]      #センサ値
            obs_id  = d[1] - 1  #ID

            #パーティクルの位置と地図からランドマークの距離と方角を算出
            pos_on_map = envmap.landmarks[obs_id].pos
            #particle_suggest_pos = IdealCamera.relative_polar_pos(self.pose, pos_on_map)
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

            #尤度の計算
            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)
            
    



'''----- パーティクルを管理するMCL(Monte Carlo localization)クラス -----'''
# envmap            : 地図のオブジェクト
# init_pose         : 初期姿勢
# num               : パーティクル数
# motion_noise_stds : ロボットの動きに生じる雑音の標準偏差
#                   : nn : 直進1[m]   で生じる 道のり のばらつきの標準偏差
#                   : no : 回転1[rad] で生じる 道のり のばらつきの標準偏差
#                   : on : 直進1[m]   で生じる   向き のばらつきの標準偏差
#                   : oo : 回転1[rad] で生じる   向き のばらつきの標準偏差
# distance_dev_rate : 距離に加える雑音の標準偏差
# direction_dev     : 方角に加える雑音の標準偏差
class Mcl:
    def __init__(self, envmap, init_pose, num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)] #パーティクルオブジェクトのリスト生成
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev     = direction_dev
        
        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)        #4次元のガウス分布オブジェクト生成

        self.ml   = self.particles[0]
        self.pose = self.ml.pose


    ### 重み最大のパーティクル(ml: maximum likelihood) を選ぶ ############################################
    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose


    ### パーティクルを動かす ##############################################################################
    # nu    : 速度
    # omega : 角速度
    # time  : 離散時間1ステップの時間
    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf) #Particleクラスのmotion_updateメソッド


    ### 観測したセンサ値を使って処理 #####################################################################
    # obsevation : センサ値
    def observation_update(self, observation):
        for p in self.particles: #Particleクラスのobservation_updateメソッドを呼び出す
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()


    ### リサンプリング ###################################################################################
    def resampling(self):
        ### リサンプリング(重みの大きさに応じて1つずつ選んでいく)
        # ws = [e.weight for e in self.particles]  #重みのリストを作成
        # if sum(ws) < 1e-100:  #重みの和がゼロに丸め込まれるとエラーになるので小さな数を足しておく
        #     ws = [e + 1e-100 for e in ws]
        #
        ## wsの要素に比例した確率でパーティクルをnum個選ぶ
        # ps = random.choices(self.particles, weights=ws, k=len(self.particles)) 
        # self.particles = [copy.deepcopy(e) for e in ps]  #選んだリストからパーティクルを取り出す
        # for p in self.particles:
        #     p.weight = 1.0/len(self.particles)           #重みの正規化

        
        ### 系統リサンプリング (こっちのほうが計算量, サンプリングバイアスが小さい)
        ws = np.cumsum([e.weight for e in self.particles]) #重みを累積して足していく(最後の要素は重みの合計)
        ## 例) a = np.array([1,2,3,4,5,6])
        ##     b = np.cumsum(a)
        ##     print(b)
        ##     >>> np.array([1,3,6,10,15,21])
        
        if ws[-1] < 1e-100:   #重みの合計がゼロにならないように処理
            ws = [e + 1e-100 for e in ws]

        step = ws[-1]/len(self.particles)    #幅を求める. (重みの合計)/(パーティクル数)
        r    = np.random.uniform(0.0, step)  #最初のパーティクルを選択

        cur_pos = 0 #要素インデックス
        ps = []     #抽出するパーティクルのリスト

        while(len(ps) < len(self.particles)): #もとのパーティクル数と同数になるまで選ぶ
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)

            
    ### パーティクルの描画 ###############################################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        #全パーティクルのX座標, Y座標をリスト化する
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]

        #向きをベクトルで表したときのX座標成分, Y座標成分をリスト化する
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for  p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for  p in self.particles]

        #quiver : 矢印を描画するメソッド
        elems.append(ax.quiver(xs, ys, vxs, vys,
                               angles='xy', scale_units='xy', scale=1.5, 
                               color="blue", alpha=0.5))





'''----- 推定するエージェントクラス -----'''
# time_interval : 離散時間1ステップの時間
# nu            : 速度
# omega         : 角速度
# estimator     : 推定器のオブジェクト
class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0


    ### エージェントの判断 ################################################################################
    # observation : センサ値
    #-----------------------------
    # RET         : 速度, 角速度
    def decision(self, observation=None):
        #1つ前の制御指令値でパーティクルの姿勢を更新する
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega


    ### 重み最大パーティクルの姿勢情報を描画 ############################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))

        
        
        


def trial(motion_noise_stds):   
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(m, initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    
    world.draw()





    
if __name__ == '__main__':
    trial({"nn":0.19, "no":0.001, "on":0.13, "oo":0.2})

    
