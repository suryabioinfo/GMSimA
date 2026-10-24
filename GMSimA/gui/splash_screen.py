import os
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtGui import QColor, QPixmap, QRegion
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QRect


class SplashScreen(QSplashScreen):

    def __init__(self):

        base_dir = os.path.dirname(os.path.dirname(__file__))
        image_path = os.path.join(base_dir, "assets", "splash2.png")

        pixmap = QPixmap(image_path)

        #pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = pixmap.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        super().__init__(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        # circular splash
        self.setMask(QRegion(0, 0, 400, 400, QRegion.Ellipse))

        # fade animation
        self.setWindowOpacity(0)

        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

        # center splash
        screen = self.screen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        # loading messages
        self.messages = [
            "Initializing GMSimA ...",
            "Preparing GUI Components ...",
            "Loading Analysis Modules ...",
            "Welcome to GMSimA !"
        ]

        self.message_index = 0

        # rectangle where text will appear
        self.text_rect = QRect(0, (self.height() - 40) // 2, self.width(), 40)

        #self.text_rect = QRect(0, int(self.height()*0.91), self.width(), 24) # at bottom of splash screen

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_message)
        self.timer.start(1000)  # update every 2 seconds
        self.update_message()
        #QTimer.singleShot(25000, self.close)


    def update_message(self):
        if self.message_index < len(self.messages):
            self.message_index += 1
            self.update()
        else:
            self.timer.stop()

    def drawContents(self, painter):
        super().drawContents(painter)

        if 0 < self.message_index <= len(self.messages):
            message = self.messages[self.message_index - 1]

            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)

            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(message)
            padding_x = 24
            padding_y = 10

            badge_width = text_width + padding_x
            badge_height = metrics.height() + padding_y

            # Centered over self.text_rect
            badge_x = self.text_rect.center().x() - (badge_width // 2)
            badge_y = self.text_rect.center().y() - (badge_height // 2)
            badge_rect = QRect(badge_x, badge_y, badge_width, badge_height)

            painter.setRenderHint(painter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.NoPen)

            # --- TRANSPARENCY SETTING ---
            # QColor(Red, Green, Blue, Alpha)
            alpha_transparency = 200  
            painter.setBrush(QColor(192, 192, 192, alpha_transparency))

            # Draw rounded pill badge
            painter.drawRoundedRect(badge_rect, 12, 12)

            # Draw red message
            painter.setPen(Qt.red)
            painter.drawText(badge_rect, Qt.AlignCenter, message)
