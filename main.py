import sys

from BlurTool import BlurTool
from PullAndCheck import PullAndCheck

blurTool = BlurTool()
pullAndCheck = PullAndCheck()

# Copy data to computer, Perform sanity checks
try:
    vidPath = pullAndCheck.pull()
    pullAndCheck.check_filesize(vidPath)
    pullAndCheck.check_timestamp(vidPath)
    blurTool.process_video(vidPath)
except Exception as exception:
    print(exception)
    sys.exit(1)






