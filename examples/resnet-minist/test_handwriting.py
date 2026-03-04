import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QLabel, QHBoxLayout
)
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtCore import Qt, QPoint
from PIL import Image
import torch
from torchvision import transforms

from models.resnet import ResNet18

class DrawBoard(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.black)
        self.setPixmap(self.pixmap)
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.white, 12, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.setPixmap(self.pixmap)

    def clear(self):
        self.pixmap.fill(Qt.black)
        self.setPixmap(self.pixmap)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST 手写识别测试")

        self.board = DrawBoard()

        self.result_label = QLabel("预测结果：")
        self.result_label.setStyleSheet("font-size: 22px;")

        self.btn_predict = QPushButton("识别")
        self.btn_clear = QPushButton("清空")

        self.btn_predict.clicked.connect(self.predict)
        self.btn_clear.clicked.connect(self.board.clear)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_predict)
        btn_layout.addWidget(self.btn_clear)

        layout = QVBoxLayout()
        layout.addWidget(self.board)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNet18(1, 10)
        self.model.load_state_dict(torch.load(
            "./checkpoints/resnet-mnist/best_model.pth",
            map_location=self.device,
            weights_only=True
        ))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

    def predict(self):
        qimage = self.board.pixmap.toImage()

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        arr = np.array(ptr).reshape(height, width, 4)

        # 转灰度
        img = Image.fromarray(arr[:, :, 0])
        img = img.convert("L")

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            pred = torch.argmax(output, dim=1).item()

        self.result_label.setText(f"预测结果：{pred}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())