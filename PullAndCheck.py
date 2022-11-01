import os
import sys
from pathlib import Path

import cv2
from adb import adb_commands, sign_m2crypto


class FileSizeTooSmallError(Exception):
    """raised when file size is too small for duration of video"""

    pass


class TimestampDurationMatchError(Exception):
    """raised when timestamp duration does not match duration of video"""

    pass

class PullAndCheck:
    def __init__(self):
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[1]

        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))

        self.data_dir_path = ROOT.joinpath("Data")

        # Enable USB Debugging mode of your mobile phone

        # Authentication and Connect to the device
        signer = sign_m2crypto.M2CryptoSigner(os.path.expanduser("~/.android/adbkey"))
        self.device = adb_commands.AdbCommands()
        self.device.ConnectDevice(rsa_keys=[signer])


    def copy_file(self, src_path, dst_path):

        data_bytes = self.device.Pull(src_path)
        with open(dst_path, "wb") as f:
            f.write(data_bytes)

    '''Returns path of mp4 video.'''
    def pull(self):
        base_dir_path = "/storage/emulated/0/DCIM/OpenCamera/"
        data_dir_path = str(self.data_dir_path)

        files = self.device.List(base_dir_path)
        for file in files:
            file_name = getattr(file, "filename").decode("utf-8")

            if "." in file_name:
                # file: jpg / mp4 / srt
                self.copy_file(base_dir_path + file_name, data_dir_path + "/" + file_name)
            else:
                # dir
                Path(data_dir_path + "/" + file_name).mkdir(exist_ok=True)

                for sub_file in self.device.List(base_dir_path + file_name):
                    sub_file_name = getattr(sub_file, "filename").decode("utf-8")
                    self.copy_file(
                        base_dir_path + file_name + "/" + sub_file_name,
                        data_dir_path + "/" + file_name + "/" + sub_file_name,
                    )

            if ".mp4" in file_name:
                vid_path = data_dir_path + "/" + file_name
        return vid_path


    def check_filesize(vid_path: str):
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps
        cap.release()
        file_byte = os.path.getsize(vid_path)
        bitrate = 20000000
        expected_size = bitrate * duration / 8
        if file_byte < expected_size * 0.8:
            raise FileSizeTooSmallError


    def check_timestamp(vid_path: str, csv_path: str):
        f = open(csv_path, "r")
        lines = f.readlines()
        f.close()
        start_timestamp, end_timestamp = lines[0], lines[-1]
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        timestamp_duration = (int(end_timestamp) - int(start_timestamp)) / pow(10, 9)
        if round(duration, 1) != round(timestamp_duration, 1):
            raise TimestampDurationMatchError


# if __name__ == "__main__":
#     pull()
#     check_filesize(
#         "/home/yangbo/Documents/Internal/Blurring/Blur/license-plate-blurring/Data/VID_20221016_143907.mp4"
#     )
#     check_timestamp(
#         "/home/yangbo/Documents/Internal/Blurring/Blur/license-plate-blurring/Data/VID_20221016_164644.mp4",
#         "/home/yangbo/Documents/Internal/Blurring/Blur/license-plate-blurring/Data/20221016_164644/VID_20221016_164644_imu_timestamps.csv",
#     )
