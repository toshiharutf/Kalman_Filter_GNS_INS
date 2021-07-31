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

GNSS
    Position (NED) : Note GNSS will normally provide position in ECEF of LLH coordinates
                     The measurements in the provided data files are already transformed to NED
                     The equations for transformation can be found in: https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
    - pos_n
    - pos_e
    - pos_d

    Velocity (NED)
    - v_n
    - v_e
    - v_d

    New Measurement flag
    - gnss_new_data

Magnetometer
    - mag_field_x
    - mag_field_y
    - mag_field_z


X: states:
    - pos_n
    - pos_e
    - pos_d
    - v_n
    - v_e
    - v_d
    - roll
    - pitch
    - yaw
    - bias_gyro_x
    - bias_gyro_y
    - bias_gyro_z
    - bias_accel_x
    - bias_accel_y
    - bias_accel_z

u:    inputs
    - Acceleration
    - Angular rates


"""

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as rot
import pandas as pd
"""
X: states:
    - error in pitch
    - error in roll
    - bias angular rate pitch
    - bias angular rate roll

    measurements:
    - specific forces
"""

class GNSS_filter:

    def __init__(self, row, mag_ref):
        dt = 1e-2
        dt2 = dt*dt
        self.mag_ref = mag_ref

        # this value must be collected with the IMU at rest
        self.g = np.sqrt(row["imu_accel_x"]**2+row["imu_accel_y"]**2+row["imu_accel_z"]**2)
        self.vector_g = np.matrix([0, 0, self.g]).transpose()

        self.Nx = 15 # number of states: pos, vel, attitude, acc bias, gyro bias
        self.Nu = 9 # number of input var: acc, angular vel, gravity (to help calc)
        self.Nm = 9 #  number of measurement vars: pos, vel

        self.X = np.zeros([self.Nx, 1]) # error states: pos_ned, v_ned, eul, bias_acc, bias_gyro
        self.X[6] = np.arctan2(row["imu_accel_y"], np.sqrt(row["imu_accel_y"]**2+row["imu_accel_z"]**2))
        self.X[7] = np.arctan2(row["imu_accel_x"], np.sqrt(row["imu_accel_x"]**2+row["imu_accel_z"]**2))
        self.X[8] = self.mag_heading(self.X[6:9],
                                      mag_ref,
                                      np.matrix([row["mag_field_x"],row["mag_field_y"],row["mag_field_z"]]))

        # Rotational matrix from body to NED frame
        self.Cnb = rot.from_euler("xyz", self.X[6:9].transpose()).as_matrix()[0]

    # Process model

        self.F = np.identity(self.Nx)
        self.F[0:3, 3:6] = np.identity(3)*dt
        self.F[0:3, 12:15] = -np.identity(3)*dt2/2
        self.F[3:6, 12:15] = -np.identity(3)*dt
        self.F[6:9, 9:12] = -np.identity(3)*dt

        self.P = np.identity(self.Nx)

        # Process noise matrix
        self.Q = np.zeros([self.Nx, self.Nx])

        # Position estimation noise
        self.Q[0:3, 0:3] = np.identity(3)*1

        # Velocity estimation noise (acc psd)
        self.Q[3:6, 3:6] = np.identity(3)*10

        # Attitude estimation noise (gyro psd)
        self.Q[6:9, 6:9] = np.identity(3)*3.5e-3

        # Acceleration bias estimation noise (acc bias psd)
        self.Q[9:12, 9:12] = np.identity(3)*1.0e-8

        # Gyro bias estimation noise (gyro bias psd)
        self.Q[12:15, 12:15] = np.identity(3)*1.0e-7


        # Position measurement noise
        self.R_position = np.identity(3)*2

        # Velocity measurement noise
        self.R_velocity = np.identity(3)

        # Attitude (acc noise)
        self.R_attitude = np.identity(2)*5

        # Heading
        self.R_heading = 0.0625


    def updateQ(self,dt):
        # Position estimation noise
        self.Q[0:3, 0:3] = np.identity(3)*1e-2*dt

        # Velocity estimation noise (acc psd)
        self.Q[3:6, 3:6] = np.identity(3)*1e-1*dt

        # Attitude estimation noise (gyro psd)
        self.Q[6:9, 6:9] = np.identity(3)*3.5e-1*dt

        # Acceleration bias estimation noise (acc bias psd)
        self.Q[9:12, 9:12] = np.identity(3)*1.0e-6*dt

        # Gyro bias estimation noise (gyro bias psd)
        self.Q[12:15, 12:15] = np.identity(3)*1.0e-5*dt


    def getEulerAnglesFromAccel(self, a_bib):
        eul_nb = np.zeros([2,1])
        eul_nb[0] = -np.arctan2(a_bib[1], np.sqrt(a_bib[1]**2+a_bib[2]**2))
        eul_nb[1] = np.arctan2(a_bib[0], np.sqrt(a_bib[0]**2+a_bib[2]**2))

        return eul_nb

    def predict(self, a_bib, w_bib, dt): # w is the angular rate vector
        self.Cnb = rot.from_euler("xyz", self.X[6:9].transpose()).as_matrix()[0]
        dt2 = dt*dt

        # Specific force fb
        a_bib = a_bib.transpose()
        fb = self.Cnb@a_bib + self.vector_g
        u = np.zeros([self.Nx, 1])
        u[0:3] = fb*dt2/2
        u[3:6] = fb*dt
        u[6:9] = w_bib.transpose()*dt


        self.F[0:3, 3:6] = np.identity(3)*dt
        self.F[0:3, 12:15] = -self.Cnb*dt2/2
        self.F[3:6, 12:15] = -self.Cnb*dt
        self.F[6:9, 9:12] = -self.Cnb*dt

        # self.updateQ(dt)

        self.X = self.F@self.X + u
        self.P = self.F@self.P@self.F.transpose() + self.Q


    def updateAttitude(self, a_bib=np.zeros([1,3])):
        H = np.zeros([2, self.Nx])
        H[0,6] = 1
        H[1,7] = 1

        eul_nb_k = self.getEulerAnglesFromAccel(a_bib.transpose())

        y = eul_nb_k - H @ self.X
        S = H @ self.P @ H.transpose() + self.R_attitude
        K = (self.P @ H.transpose()) @ inv(S)
        self.X += K @ y

        I = np.identity(self.Nx)
        self.P = (I - K @ H) @ self.P


    def updateHeading(self, mag_field= np.zeros([1,3]) ):
        H = np.zeros([1, self.Nx])
        H[0,8] = 1
        mag_heading = self.mag_heading(self.X[6:9], self.mag_ref, mag_field)

        y = mag_heading - H @ self.X
        S = H @ self.P @ H.transpose() + self.R_heading
        K = (self.P @ H.transpose()) @ inv(S)
        self.X += K @ y

        I = np.identity(self.Nx)
        self.P = (I - K @ H) @ self.P

    def updatePosition(self, pos_ned= np.zeros([1,3])):
        H = np.zeros([3, self.Nx])
        H[0:3, 0:3] = np.identity(3)

        y = pos_ned.transpose() - H @ self.X
        S = H @ self.P @ H.transpose() + self.R_position
        K = (self.P @ H.transpose()) @ inv(S)
        self.X += K @ y

        I = np.identity(self.Nx)
        self.P = (I - K @ H) @ self.P

    def updateVelocity(self, v_ned= np.zeros([1,3])):
        H = np.zeros([3, self.Nx])
        H[0:3,3:6] = np.identity(3)

        y = v_ned.transpose() - H @ self.X
        S = H @ self.P @ H.transpose() + self.R_velocity
        K = (self.P @ H.transpose()) @ inv(S)
        self.X += K @ y

        I = np.identity(self.Nx)
        self.P = (I - K @ H) @ self.P


    # z is the height measured
    def update(self, a_bib=np.zeros([1,3]), pos_ned= np.zeros([1,3]), v_ned= np.zeros([1,3]), mag_field= np.zeros([1,3]) ):
        self.Cnb = rot.from_euler("xyz", self.X[6:9].transpose()).as_matrix()[0]

        z = np.zeros([self.Nm,1])
        eul_nb_k = self.X[6:9]
        pos_ned_k = self.X[0:3]
        v_ned_k = self.X[3:6]

        if a_bib.any():
            eul_nb_k = self.getEulerAnglesFromAccel(a_bib.transpose())
            eul_nb_k[2] = self.X[8]

        if mag_field.any():
            eul_nb_k[2] = self.mag_heading(eul_nb_k, self.mag_ref, mag_field)

        if pos_ned.any():
            pos_ned_k = pos_ned.transpose()

        if v_ned.any():
            v_ned_k = v_ned.transpose()

        z[0:3] = pos_ned_k
        z[3:6] = v_ned_k
        z[6:9] = eul_nb_k


        y = z - self.H@self.X
        S = self.H@self.P@self.H.transpose() + self.R
        K = (self.P@self.H.transpose())@inv(S)
        self.X += K@y

        I = np.identity(self.Nx)
        self.P = (I-K@self.H)@self.P


    def mag_heading(self, eul_nb, local_mag_ref, mag_field_norm):
        mag_field_norm = mag_field_norm.transpose()

        # copy value that has been passed by reference, to avoid overwriting it.
        temp_eul = np.zeros([1,3])
        temp_eul[0,0] = eul_nb[0]
        temp_eul[0,1] = eul_nb[1]

        Cnb = rot.from_euler("xyz", temp_eul).as_matrix()[0]
        mag_field_nf = Cnb@mag_field_norm
        heading = -np.arctan2(mag_field_nf[1], mag_field_nf[0]) + np.arctan2(local_mag_ref[1], local_mag_ref[0])

        return np.asscalar(heading)


    def get_states(self):
        return {
                "pos_n": np.asscalar(self.X[0]),
                "pos_e": np.asscalar(self.X[1]),
                "pos_d": np.asscalar(self.X[2]),
                "v_n": np.asscalar(self.X[3]),
                "v_e": np.asscalar(self.X[4]),
                "v_d": np.asscalar(self.X[5]),
                "roll": np.asscalar(self.X[6])*180/np.pi,
                "pitch": np.asscalar(self.X[7])*180/np.pi,
                "yaw": np.asscalar(self.X[8])*180/np.pi,
                "gyro_bias_roll": np.asscalar(self.X[9])*180/np.pi,
                "gyro_bias_pitch": np.asscalar(self.X[10])*180/np.pi,
                "gyro_bias_yaw": np.asscalar(self.X[11])*180/np.pi,
                "acc_bias_x": np.asscalar(self.X[12]),
                "acc_bias_y": np.asscalar(self.X[13]),
                "acc_bias_z": np.asscalar(self.X[14])
        }


def run_filter_simulation(df, mag_ref):
    import time

    start = time.time()

    init = False
    results = pd.DataFrame()

    last_time = 0
    dt = 0
    for index, row in df.iterrows():

        if not init:
            gnss_kf = GNSS_filter(row, mag_ref)
            init = True
            dt = 1e-2


        gnss_kf.predict(np.matrix([row["imu_accel_x"], row["imu_accel_y"], row["imu_accel_z"]]),
                        np.matrix([row["imu_gyro_x"], row["imu_gyro_y"], row["imu_gyro_z"]]), dt)

        #Attitude update happens at each step
        # Note: The IMU run faster than the data collection device used (100 Hz). Therefore, all measurements
        # from the IMU are always considered as new measurements.

        gnss_kf.updateAttitude(a_bib = np.matrix([row["imu_accel_x"], row["imu_accel_y"], row["imu_accel_z"]]))

        # new magnetometer measurement
        if row["mag_new_data"]:
            gnss_kf.updateHeading(mag_field = np.matrix([row["mag_field_x"],row["mag_field_y"],row["mag_field_z"]]) )

        # new GNSS measurement
        if row["gnss_new_data"]:
            gnss_kf.updatePosition(pos_ned = np.matrix([row["pos_n"], row["pos_e"], row["pos_d"]]))
            gnss_kf.updateVelocity(v_ned = np.matrix([row["v_n"], row["v_e"], row["v_d"]]))

        dt = row["time"] - last_time
        last_time = row["time"]

        res = gnss_kf.get_states()
        res["time"] = row["time"]
        results = results.append(res, ignore_index=True)


    end = time.time()
    print(f"Execution time: {end - start} s")

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(5, 3)
    ax[0,0].set_title("Roll X")
    ax[0,0].plot(results["time"], results["roll"], label="roll")

    ax[0,1].set_title("Roll Y")
    ax[0,1].plot(results["time"], results["pitch"], label="pitch")

    ax[0,2].set_title("Roll Z")
    ax[0,2].plot(results["time"], results["yaw"], label="yaw")

    ax[1,0].set_title("Bias Gyro X")
    ax[1,0].plot(results["time"], results["gyro_bias_roll"], label="gyro_bias_roll")

    ax[1,1].set_title("Bias Gyro Y")
    ax[1,1].plot(results["time"], results["gyro_bias_pitch"], label="gyro_bias_pitch")

    ax[1,2].set_title("Bias Gyro Z")
    ax[1,2].plot(results["time"], results["gyro_bias_yaw"], label="gyro_bias_yaw")

    ax[2,0].set_title("Pos NED X")
    ax[2,0].plot(results["time"], results["pos_n"], label="pos N")

    ax[2,1].set_title("Pos NED Y")
    ax[2,1].plot(results["time"], results["pos_e"], label="pos E")

    ax[2,2].set_title("Pos NED Z")
    ax[2,2].plot(results["time"], results["pos_d"], label="pos D")

    ax[3,0].set_title("V NED X")
    ax[3,0].plot(results["time"], results["v_n"], label="v N")
    ax[3,0].plot(results["time"], df["v_n"], linestyle="dotted", label="V N")

    ax[3,1].set_title("V NED Y")
    ax[3,1].plot(results["time"], results["v_e"], label="v E")
    ax[3,1].plot(results["time"], df["v_e"], linestyle="dotted", label="V E")

    ax[3,2].set_title("V NED Z")
    ax[3,2].plot(results["time"], results["v_d"], label="v D")
    ax[3,2].plot(results["time"], df["v_d"], linestyle="dotted", label="V D")

    ax[4,0].set_title("Bias Acc X")
    ax[4,0].plot(results["time"], results["acc_bias_x"], label="bias_acc_0")

    ax[4,1].set_title("Bias Acc Y")
    ax[4,1].plot(results["time"], results["acc_bias_y"], label="bias_acc_1")

    ax[4,2].set_title("Bias Acc Z")
    ax[4,2].plot(results["time"], results["acc_bias_z"], label="bias_acc_2")

    # f.legend()
    # f.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    f.canvas.set_window_title('Kalman Filter GNS/INS')
    f.suptitle("Kalman Filter GNS/INS")
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("gns_ins_data.csv")
    # Magnetic field reference based on geo-location from the World Magnetic Model
    # more information in http://www.ngdc.noaa.gov/geomag/WMM/
    mag_ref = [19.643034, 0.79634374, 44.803173]

    run_filter_simulation(data, mag_ref)