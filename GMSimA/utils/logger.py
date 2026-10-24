import os
from datetime import datetime


class SmgroanaLogger:

    def __init__(self, output_dir):

        self.logfile = os.path.join(output_dir, "GMSimA.log")

        # load ascii logo
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets",
            "logimage.txt"
        )

        with open(logo_path, "r", encoding="utf8") as f:
            logo = f.read()

        with open(self.logfile, "w", encoding="utf8") as log:

            log.write(logo)
            log.write("\n\n")
            log.write("GMSimA Run Log\n")
            log.write(f"Date: {datetime.now()}\n")
            log.write("_____________________________________________________________________________________________________________\n\n")

    def write(self, message):
        with open(self.logfile, "a", encoding="utf8") as log:

            log.write(message + "\n")