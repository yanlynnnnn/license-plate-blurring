import numpy as np
import os
import pandas as pd
np.random.seed(20)

class Stats:
    # action: left turn/right turn
    # start/end: decimal from 0 to 1
    # date: unique identifier of each driving video
    def get_avg_num_frames(self, action: int, start, end, date):
        frame_count = 0
        count = 0
        arr = np.load('../Datasets/hdd_data/target/%s.npy' % date, allow_pickle=True)
        target_arr = arr[int(start*len(arr)):int(end*len(arr))]
        for j in range(len(target_arr)):
            if target_arr[j] == action:
                frame_count += 1
            if target_arr[j] == action and target_arr[j-1] != action:
                count += 1
        if count != 0:
            return round(frame_count/count, 3)
        else:
            return None

    # stat: 0 - max, 1 - avg, 2 - var
    # sensor: 0 for acceleration, 7 for yaw rate
    # action: left turn/right turn etc
    def calc_sensor_data(self, action: int, stat: int, sensor: int, start, end, date):
        assert stat in [0, 1, 2], "Select 0 for max, 1 for avg, 2 for var"
        assert sensor >= 0 and sensor <= 7, "Select 0 for acceleration or 7 for yaw rate"
        target_arr = np.load('../Datasets/hdd_data/target/%s.npy' % date, allow_pickle=True)
        sensor_arr = np.load('../Datasets/hdd_data/sensor/%s.npy' % date, allow_pickle=True)
        target_arr, sensor_arr = target_arr[int(start*len(target_arr)):int(end*len(target_arr))], sensor_arr[int(start*len(sensor_arr)):int(end*len(sensor_arr))]
        assert len(target_arr) == len(sensor_arr), "Array length mismatch"
        sensor_acc = []
        for i in range(len(target_arr)):
            if target_arr[i] == action:
                sensor_acc.append(sensor_arr[i][sensor])
        if len(sensor_acc):
            if stat == 0:
                return np.max(sensor_acc)
            elif stat == 1:
                return np.mean(sensor_acc)
            else:
                return np.var(sensor_acc)
        else:
            return None

class Splitter:
    def __init__(self):
        self.target = os.listdir('../Datasets/hdd_data/target')
        self.target.remove('README.md')
        self.target.remove('.README.md.swp')
        self.target.sort()
        self.sensors = os.listdir('../Datasets/hdd_data/sensor')
        self.sensors.remove('README.md')
        self.sensors.remove('.README.md.swp')
        self.sensors.sort()
        self.imgs = os.listdir('../Datasets/hdd_data/camera')
        self.imgs.remove('.DS_Store')
        self.imgs.sort()
        self.stats = Stats()

    # each driver takes 10 elements in sequence 
    def generate_avg_num_frames(self):
        result_arr = [] 
        for date in self.imgs:
            starts = (x * 0.1 for x in range(0, 10))
            for start in starts:
                action_arr = []
                for action in [2, 3, 10, 11]:
                    avg_num_frames = self.stats.get_avg_num_frames(action, start, (start+0.1), date)
                    action_arr.append(avg_num_frames)
                result_arr.append(action_arr)
        return np.array(result_arr)

    def generate_sensor_data(self):
        result_arr = []
        for date in self.imgs:
            starts = (x * 0.1 for x in range(0, 10))
            for start in starts:
                action_arr = []
                stats = [0, 1, 2]
                sensors = [0, 7]
                actions = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
                for action in actions:
                    for sensor in sensors:
                        for stat in stats:
                            data = self.stats.calc_sensor_data(action, stat, sensor, start, (start+0.1), date)
                            action_arr.append(data)
                result_arr.append(action_arr)
        return np.array(result_arr)

    def generate_features(self):
        num_frames = self.generate_avg_num_frames()
        sensor_data = self.generate_sensor_data()
        combined = np.hstack((num_frames, sensor_data))
        columns = ['AvgFramesLeftTurn', 'AvgFramesRightTurn', 'AvgFramesMerge', 'AvgFramesUTurn',
                    'IntersectionMaxAcc', 'IntersectionAvgAcc', 'IntersectionVarAcc', 'IntersectionMaxYaw',
                    'IntersectionAvgYaw', 'IntersectionVarYaw', 'LeftTurnMaxAcc', 'LeftTurnAvgAcc', 
                    'LeftTurnVarAcc', 'LeftTurnMaxYaw', 'LeftTurnAvgYaw', 'LeftTurnVarYaw', 'RightTurnMaxAcc', 
                    'RightTurnAvgAcc', 'RightTurnVarAcc', 'RightTurnMaxYaw', 'RightTurnAvgYaw', 'RightTurnVarYaw',
                    'LeftLaneMaxAcc', 'LeftLaneAvgAcc', 'LeftLaneVarAcc', 'LeftLaneMaxYaw', 'LeftLaneAvgYaw', 
                    'LeftLaneVarYaw', 'RightLaneMaxAcc', 'RightLaneAvgAcc', 'RightLaneVarAcc', 'RightLaneMaxYaw', 'RightLaneAvgYaw', 
                    'RightLaneVarYaw', 'LeftBranchMaxAcc', 'LeftBranchAvgAcc', 'LeftBranchVarAcc', 'LeftBranchMaxYaw', 'LeftBranchAvgYaw', 
                    'LeftBranchVarYaw', 'RightBranchMaxAcc', 'RightBranchAvgAcc', 'RightBranchVarAcc', 'RightBranchMaxYaw', 'RightBranchAvgYaw', 
                    'RightBranchVarYaw', 'CrosswalkMaxAcc', 'CrosswalkAvgAcc', 'CrosswalkVarAcc', 'CrosswalkMaxYaw', 'CrosswalkAvgYaw', 
                    'CrosswalkVarYaw', 'MergeMaxAcc', 'MergeAvgAcc', 'MergeVarAcc', 'MergeMaxYaw', 'MergeAvgYaw', 
                    'MergeVarYaw', 'UTurnMaxAcc', 'UTurnAvgAcc', 'UTurnVarAcc', 'UTurnMaxYaw', 'UTurnAvgYaw', 
                    'UTurnVarYaw']

        df = pd.DataFrame(combined, columns=columns)
        dates = np.repeat(np.array(self.imgs), 10)
        df['Date'] = dates
        return df


        




    