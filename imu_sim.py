import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as rot

class IMU:
    def __init__(self, acc_bias=None,
                 acc_rw=None,
                 gyro_bias=None,
                 gyro_rw=None):

        if acc_bias is None:
            acc_bias = [0, 0, 0]
        if acc_rw is None:
            acc_rw = [0, 0, 0]
        if gyro_bias is None:
            gyro_bias = [0, 0, 0]
        if gyro_rw is None:
            gyro_rw = [0, 0, 0]

        self.acc_bias = acc_bias
        self.acc_rw = acc_rw
        self.gyro_bias = gyro_bias
        self.gyro_rw = gyro_rw

        self.g = np.matrix([0,0,-9.81]).transpose()

    def run_sim(self, total_time=100, dt=0.01, attAmplitude=np.pi / 6, attPeriod=2):
        time = np.arange(0, total_time, dt)
        roll_ideal = attAmplitude * np.sin(2 * np.pi / attPeriod * time)
        w_roll_ideal = attAmplitude * (2 * np.pi / attPeriod) * np.cos(2 * np.pi / attPeriod * (time + dt))

        pitch_ideal = attAmplitude * np.cos(2 * np.pi / attPeriod * time)
        w_pitch_ideal = -attAmplitude * (2 * np.pi / attPeriod) * np.sin(2 * np.pi / attPeriod * (time + dt))

        a_bib = np.zeros([3, time.shape[0]])
        w_bib = np.zeros([3, time.shape[0]])
        for i in np.arange(0, time.shape[0]):
            Cnb = rot.from_euler("xy", [roll_ideal[i], pitch_ideal[i]]).as_matrix()
            w_i = np.matrix([w_roll_ideal[i], w_pitch_ideal[i], 0]).transpose()
            temp = Cnb.transpose()@self.g
            temp2 = Cnb.transpose()@w_i
            a_bib[0,i] = temp[0]
            a_bib[1, i] = temp[1]
            a_bib[2, i] = temp[2]
            w_bib[0, i] = temp2[0]
            w_bib[1, i] = temp2[1]
            w_bib[2, i] = temp2[2]

        # Add random walk + bias
        for i in range(0,3):
            a_bib[i, :] += self.acc_rw[i] * np.random.rand(time.shape[0]) + self.acc_bias[i]
                           # * np.cos(2 * np.pi / 3000 * time)

            w_bib[i, :] += self.gyro_rw[i] * np.random.rand(time.shape[0]) + self.gyro_bias[i]
                           # * np.cos(2 * np.pi / 3000 * time)

        # simulate flag of new IMU measurement
        imu_new_data = np.ones(time.shape[0])

        return {"time": time,
                "roll_ideal": roll_ideal*180/np.pi,
                "pitch_ideal": pitch_ideal*180/np.pi,
                "imu_accel_x": a_bib[0,:],
                "imu_accel_y": a_bib[1, :],
                "imu_accel_z": a_bib[2, :],
                "imu_gyro_x": w_bib[0,:],
                "imu_gyro_y": w_bib[1,:],
                "imu_gyro_z": w_bib[2,:],
                "imu_new_data": imu_new_data}


if __name__ == "__main__":
    imu = IMU(acc_bias=[0,0,0], acc_rw=[4,4,0.05], gyro_bias=[3.5e-4,3.5e-4,3.5e-4], gyro_rw=[5e-1,5e-1,5e-1])
    data = imu.run_sim(total_time=30)
    df = pd.DataFrame(data)
    df.to_csv("imu_sim.csv")

    f, ax = plt.subplots(5, 1)
    ax[0].plot(data["time"], data["roll_ideal"])
    ax[1].plot(data["time"], data["imu_accel_x"])
    ax[2].plot(data["time"], data["imu_accel_y"])
    ax[3].plot(data["time"], data["imu_accel_z"])
    ax[4].plot(data["time"], data["imu_gyro_x"])

    plt.show()
