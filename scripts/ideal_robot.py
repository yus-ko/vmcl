#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

# 理想的なロボット
# バイアスなど無し

import matplotlib
#matplotlib.use('nbagg') #notebook用?
import matplotlib.animation as anm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math




'''----- 世界座標系上にあるものの管理と描画を行うクラス -----'''
# time_span     : 何秒間シミュレーションするか
# time_interval : 離散時間1ステップが何秒であるか
# debug         : Trueのとき描画しない
class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval


    ### オブジェクトを追加する #########################################################################
    # obj : オブジェクト
    def append(self, obj):
        self.objects.append(obj)


    ### 描画する #######################################################################################
    def draw(self):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        elems = []

        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig,                                             # 図のオブジェクト
                self.one_step,                                   # 1ステップ時刻を進めるメソッド
                fargs=(elems,ax),                                # one_stepに渡す引数
                frames=int(self.time_span/self.time_interval)+1, # 描画する総ステップ数
                interval=int(self.time_interval*1000),           # ステップの周期(単位:ms)
                repeat=False                                     # 繰り返し再生するか
            )
            plt.show()


    ### アニメーションを1コマ進める. (離散時間を1ステップ進める) #####################################
    # i     : ステップ番号
    # elems : 描画する図形のリスト
    # ax    : サブプロット
    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = "t=%.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): # hasattr : オブジェクトにメソッドがあるかどうかを調べる関数
                obj.one_step(self.time_interval)


                


'''----- 理想的なロボットのクラス -----'''
# pose   : 姿勢 ([x, y, theta])
# agent  : エージェント
# sensor :
# color  : 描画するときの色
class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose   = pose 
        self.r      = 0.2  #半径
        self.color  = color
        self.agent  = agent
        self.poses  = [pose]
        self.sensor = sensor

        
    ### 描画するロボットを登録する ####################################################################
    # ax    : サブプロット          どの図に
    # elems : 描画する図形のリスト  どこになにを?    
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta) #ロボットの鼻先のx座標
        yn = y + self.r * math.sin(theta) #ロボットの鼻先のy座標
        elems += ax.plot([x,xn], [y,yn], color=self.color) #直線描画 
        c = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black") #軌跡の描画
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2]) #self.poses[-2]:センサ値を得た時刻の姿勢
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)


    ### 状態遷移関数 #################################################################################
    # nu    : 速度
    # omega : 角速度
    # time  : デルタt
    # pose  : 姿勢
    #--------------------------------------------------
    # RET   : 状態遷移後の姿勢 np.array([x, y, theta])
    '''オブジェクトを作らなくても実行できる'''
    @classmethod 
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0),
                                    nu*math.sin(t0),
                                    omega]) * time
        else:
            return pose + np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)),
                                    nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)),
                                    omega*time])


    ### アニメーションを1コマ進める. (離散時間を1時刻進める) ##########################################
    # time_interval : 離散時間1ステップが何秒であるか
    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)   #エージェントに観測結果を渡す
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)




        

'''----- ロボットの制御入力を決めるエージェントのクラス -----'''
# nu    : 速度
# omega : 角速度
class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega






'''----- ランドマークを表すクラス -----'''
# x : ランドマークのx座標
# y : ランドマークのy座標
class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None


    ### ランドマークを描画する #########################################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        # s : 点のサイズ (幅の2乗で指定 [単位:ポイント])
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange") # 散布図
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))




        
'''----- 地図のクラス -----'''
class Map:
    def __init__(self):
        self.landmarks = []


    ### ランドマークを追加 #############################################################################
    # landmark : ランドマーク
    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks) + 1
        self.landmarks.append(landmark)


    ### 描画 ###########################################################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)
    



            
'''----- 理想的なカメラ -----'''            
# env_map         : 地図のオブジェクト
# distance_range  : 計測可能距離
# direction_range : 計測可能角度
class IdealCamera:
    def __init__(self, env_map, distance_range=(0.5,6.0), direction_range=(-math.pi/3,math.pi/3)):
        self.map = env_map
        self.lastdata = []
        self.distance_range = distance_range
        self.direction_range = direction_range


    ### センサ値が計測範囲に収まっているか判定する
    # polarpos : センサ値
    #----------------------------
    # RET      : 計測可能であればTrue, そうでなければFalse 
    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]


    ### すべてのランドマークの観測結果を返す ###########################################################
    # cam_pose : カメラの姿勢
    #--------------------------------------------
    # observed : すべてのランドマークの観測結果
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)

            if self.visible(z):  #センサ値が計測可能範囲内であれば追加する
                observed.append((z, lm.id))
            
        self.lastdata = observed #最後に計測したときの結果を参照できるようにしておく(draw()で使用する)
        return observed


    ### 観測関数 ######################################################################################
    # cam_pose : カメラ(ロボット)の姿勢
    # obj_pos  : 物体(ランドマーク)の姿勢
    #------------------------------------
    # RET      : 観測物体の相対位置  np.array([距離, 向き(反時計回り正)]).T
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi         #[-pi,pi)となるように正規化  
        while phi < -np.pi: phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T  #np.hypot(x,y) : sqrt(x^2 + y^2)を計算する


    ### センサ値を描画する (センサ値から世界座標系でのランドマークの位置を計算) ######################
    # ax       : サブプロット
    # elems    : 描画する図形のリスト
    # cam_pose : カメラの姿勢    
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance*math.cos(direction+theta)
            ly = y + distance*math.sin(direction+theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")




def main():            
    world = World(30, 1)
    
    m = Map()
    m.append_landmark(Landmark(2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)
    
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot1 = IdealRobot(np.array([2,3,math.pi/6]).T, sensor=IdealCamera(m), agent=straight)
    robot2 = IdealRobot(np.array([-2,-1,math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(robot1)
    world.append(robot2)
    
    cam = IdealCamera(m)

    world.draw()
    

            
if __name__ == '__main__':
    main()        






