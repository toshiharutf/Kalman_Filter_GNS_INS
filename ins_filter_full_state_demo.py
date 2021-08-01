
"""
IMU 6-DOF

    Acceleration
 - imu_accel_x
 - imu_accel_y
 - imu_accel_z

    Angular speed
 - imu_gyro_x
 - imu_gyro_y
 - imu_gyro_z

"""

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as rot
"""
X: states:
    - pitch
    - roll
    - bias angular rate pitch
    - bias angular rate roll
    
u:    inputs
    - Euler angles
"""

class INS_filter:

    def __init__(self, data):
        dt = 1e-2

        self.X = np.zeros([6,1]) # error in Euler angles, gyro biases
        self.X[0] = -np.arctan2(data["imu_accel_y"], np.sqrt(data["imu_accel_y"]**2+data["imu_accel_z"]**2))
        self.X[1] = np.arctan2(data["imu_accel_x"], np.sqrt(data["imu_accel_x"]**2+data["imu_accel_z"]**2))

        self.Cnb = rot.from_euler("xyz", self.X[0:3].transpose()).as_matrix()[0]

        self.P = np.identity(6)

        # Process model
        self.F = np.identity(6)
        self.F[0:3,3:6] = -dt*np.identity(3)

        # Control action model
        self.B = np.zeros([6,3])
        self.B[0:3, 0:3] = np.identity(3)*dt

        # Observation matrix
        self.H = np.zeros([3,6])
        self.H[0:3, 0:3] = np.identity(3)

        # Process noise matrix
        self.gyro_psd = 3.5e-4
        self.gyro_bias_psd = 1e-7

        self.Q = np.zeros([6,6])
        self.updateQ(dt)

        # Sensor noise matrix (accel)
        self.R = np.zeros([3,3])
        self.R[0][0] = 5
        self.R[1][1] = 5
        self.R[2][2] = 5

    def updateQ(self, dt):
        self.Q[0:3, 0:3] = np.identity(3)*self.gyro_psd*dt
        self.Q[3:6, 3:6] = np.identity(3) * self.gyro_bias_psd * dt

    def predict(self, w, dt): # w is the angular rate vector
        self.Cnb = rot.from_euler("xyz", self.X[0:3].transpose()).as_matrix()[0]
        u = w.transpose()

        self.updateQ(dt)

        #update dt
        self.F[0:3,3:6] = -dt*self.Cnb
        self.B[0:3, 0:3] = dt*self.Cnb

        # build pseudo control var u
        self.X = self.F@self.X + self.B@u
        self.P = self.F@self.P@self.F.transpose() + self.Q


    def updateAttitude(self, a_bib):

        z = self.getEulerAnglesFromAccel(a_bib.transpose())
        y = z - self.H@self.X
        S = self.H@self.P@self.H.transpose() + self.R
        K = (self.P@self.H.transpose())@inv(S)
        self.X = self.X+K@y

        I = np.identity(6)
        self.P = (I-K@self.H)@self.P

    def getEulerAnglesFromAccel(self, a_bib):
        eul_nb = np.zeros([3,1])
        eul_nb[0] = -np.arctan2(a_bib[1], np.sqrt(a_bib[1]**2+a_bib[2]**2))
        eul_nb[1] = np.arctan2(a_bib[0], np.sqrt(a_bib[0]**2+a_bib[2]**2))

        return eul_nb

    def get_states(self):
        return {"roll": np.asscalar(self.X[0])*180/np.pi,
                "pitch": np.asscalar(self.X[1])*180/np.pi,
                "yaw": np.asscalar(self.X[2])*180/np.pi,
                "gyro_bias_roll": np.asscalar(self.X[3])*180/np.pi,
                "gyro_bias_pitch": np.asscalar(self.X[4])*180/np.pi}

    
def run_filter_simulation(df):
    import time

    start = time.time()

    init = False
    kf_ins_res = {"roll": [], "pitch":[], "yaw":[], "gyro_bias_roll":[], "gyro_bias_pitch":[]}
    last_time = 0
    for index, row in df.iterrows():
        if not init:
            ins_kf = INS_filter(row)
            init = True
            last_time = row["time"] - 1e-2

        # Note: in a real-time system, the prediction step should run at each iteration
        # This hack is only used here for simulation purposes
        if row["imu_new_data"]:
            dt = row["time"] - last_time
            ins_kf.predict(np.matrix([row["imu_gyro_x"], row["imu_gyro_y"], row["imu_gyro_z"]]), dt)
            last_time = row["time"]

        if row["imu_new_data"]:
            ins_kf.updateAttitude(np.matrix([row["imu_accel_x"], row["imu_accel_y"], row["imu_accel_z"]]))

        res = ins_kf.get_states()
        kf_ins_res["roll"].append(res["roll"])
        kf_ins_res["pitch"].append(res["pitch"])
        kf_ins_res["yaw"].append(res["yaw"])
        kf_ins_res["gyro_bias_roll"].append(res["gyro_bias_roll"])
        kf_ins_res["gyro_bias_pitch"].append(res["gyro_bias_pitch"])


    end = time.time()
    print(f"Execution time: {end - start} s")

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(4, 1)

    ax[0].set_title("Roll")
    ax[0].plot(df["time"], kf_ins_res["roll"], label="roll")

    ax[1].set_title("Pitch")
    ax[1].plot(df["time"], kf_ins_res["pitch"], label="pitch")

    ax[2].set_title("Gyro bias roll")
    ax[2].plot(df["time"], kf_ins_res["gyro_bias_roll"], label="gyro_bias_roll")

    ax[3].set_title("Gyro bias pitch")
    ax[3].plot(df["time"], kf_ins_res["gyro_bias_pitch"], label="gyro_bias_pitch")

    plt.subplots_adjust(hspace=0.4)
    f.canvas.set_window_title('Kalman Filter INS')
    f.suptitle("Kalman Filter INS")
    # f.legend()
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("gns_ins_data2.csv")
    run_filter_simulation(data)