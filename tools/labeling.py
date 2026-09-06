#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labeling.py - 电离图描迹手动标注工具

依赖: pip install PyQt5

用法:
    python tools/labeling.py              # 弹出选文件夹对话框
    python tools/labeling.py <文件夹>     # 直接打开文件夹

功能:
- 加载文件夹下所有图片
- 左键戳点, 自动连线形成描迹(polyline)
- 右键或按钮完成当前描迹, 开始新的
- 支持多条描迹(每条带 label, 如 F2_O / F2_X)
- 保存为 JSON(与图片同名, 存于目标文件夹), 格式与 VLM 流程一致:
    {"polylines": [{"label": "F2_O", "points": [[x,y],...]}], "total_points": N}
- 切换图片自动保存; 已有 JSON 可加载继续编辑

快捷键:
    左键            戳点
    右键            完成当前描迹
    Ctrl+Z          撤销最后一点
    Ctrl+Shift+Z    撤销最后一条描迹
    Ctrl+S          保存
    Left / Right    上一张 / 下一张
    空格+左键拖动   平移视图
    滚轮            缩放
    F               适应窗口
"""

import sys
import os
import json

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainter, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QStatusBar, QSplitter, QComboBox,
    QShortcut, QGraphicsView, QGraphicsScene,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 描迹颜色(与 VLM 流程一致: 红=O, 绿=X, ...)
TRACE_COLORS = [
    QColor(255, 60, 60),
    QColor(60, 220, 60),
    QColor(60, 140, 255),
    QColor(255, 165, 0),
    QColor(180, 80, 220),
    QColor(0, 200, 200),
]

DEFAULT_LABELS = ["F2_O", "F2_X", "F1_O", "F1_X", "F_main", "E", "Es"]


class ImageCanvas(QGraphicsView):
    """画布: 显示图片 + 戳点 + 连线. scene 坐标 = 原图像素坐标."""
    pointClicked = pyqtSignal(QPointF)
    rightClicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        self.setBackgroundBrush(QColor(30, 30, 30))

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._img_rect = QRectF()
        self._trace_items = []  # 绘制的描迹图元(点/线)

    def set_image(self, path: str):
        for it in self._trace_items:
            self._scene.removeItem(it)
        self._trace_items.clear()
        self._scene.clear()
        self._pixmap_item = None

        pm = QPixmap(path)
        if pm.isNull():
            print(f"[labeling] QPixmap 加载失败: {path}", flush=True)
            self._pixmap_item = None
            self._img_rect = QRectF()
            return
        print(f"[labeling] 加载成功: {path}  {pm.width()}x{pm.height()}", flush=True)
        self._pixmap_item = self._scene.addPixmap(pm)
        self._img_rect = QRectF(0, 0, pm.width(), pm.height())
        self._scene.setSceneRect(self._img_rect)

    def fit(self):
        if self._img_rect.isNull() or self._pixmap_item is None:
            return
        self.resetTransform()
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 1 or vh <= 1:
            return
        s = min(vw / self._img_rect.width(), vh / self._img_rect.height())
        self.scale(s, s)
        self.centerOn(self._img_rect.center())

    def draw_traces(self, finished, current):
        """重绘所有描迹. finished: 已完成列表; current: 正在画的(可 None)."""
        for it in self._trace_items:
            self._scene.removeItem(it)
        self._trace_items.clear()

        polys = list(finished)
        if current and current.get("points"):
            polys.append(current)

        for i, poly in enumerate(polys):
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            pts = poly.get("points", [])
            is_current = (poly is current)
            pen_w = 2
            # 连线
            if len(pts) >= 2:
                pen = QPen(color, pen_w)
                pen.setCosmetic(True)
                for j in range(len(pts) - 1):
                    x1, y1 = pts[j]
                    x2, y2 = pts[j + 1]
                    line = self._scene.addLine(x1, y1, x2, y2, pen)
                    line.setZValue(2)
                    self._trace_items.append(line)
            # 点
            r = 4
            for k, (x, y) in enumerate(pts):
                is_last = is_current and (k == len(pts) - 1)
                pen = QPen(QColor(255, 255, 255), 1)
                ell = self._scene.addEllipse(x - r, y - r, r * 2, r * 2, pen, color)
                ell.setZValue(3)
                self._trace_items.append(ell)
                if is_last:
                    outer = self._scene.addEllipse(
                        x - r - 2, y - r - 2, (r + 2) * 2, (r + 2) * 2,
                        QPen(QColor(255, 255, 255), 1))
                    outer.setZValue(3)
                    self._trace_items.append(outer)

    def mousePressEvent(self, event):
        btn = event.button()
        if btn == Qt.LeftButton and self.dragMode() == QGraphicsView.NoDrag:
            if self._pixmap_item is not None:
                pos = self.mapToScene(event.pos())
                if self._img_rect.contains(pos):
                    self.pointClicked.emit(pos)
                    return
        elif btn == Qt.RightButton:
            self.rightClicked.emit()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    def __init__(self, folder: str = ""):
        super().__init__()
        self.setWindowTitle("电离图描迹标注")
        self.resize(1400, 900)
        screen = QApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.move(max(0, (g.width() - self.width()) // 2),
                      max(0, (g.height() - self.height()) // 2))

        self.folder = folder
        self.images = []
        self.index = -1
        self.traces = []          # [{"label":..., "points":[[x,y],...]}]
        self.current_poly = None  # 正在戳的描迹
        self.dirty = False

        self.canvas = ImageCanvas()
        self.canvas.pointClicked.connect(self.on_point)
        self.canvas.rightClicked.connect(self.finish_poly)

        # ---- 右侧控制面板 ----
        right = QWidget()
        rl = QVBoxLayout(right)

        self.btn_open = QPushButton("选择文件夹")
        self.btn_open.clicked.connect(self.choose_folder)
        rl.addWidget(self.btn_open)

        self.lbl_folder = QLabel("未选择")
        self.lbl_folder.setWordWrap(True)
        rl.addWidget(self.lbl_folder)

        self.list_imgs = QListWidget()
        self.list_imgs.currentRowChanged.connect(self.on_img_selected)
        rl.addWidget(self.list_imgs, 1)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Label:"))
        self.combo_label = QComboBox()
        self.combo_label.setEditable(True)
        self.combo_label.addItems(DEFAULT_LABELS)
        row1.addWidget(self.combo_label, 1)
        rl.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_undo_pt = QPushButton("撤销点\nCtrl+Z")
        self.btn_undo_pt.clicked.connect(self.undo_point)
        self.btn_finish = QPushButton("完成描迹\n右键")
        self.btn_finish.clicked.connect(self.finish_poly)
        row2.addWidget(self.btn_undo_pt)
        row2.addWidget(self.btn_finish)
        rl.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_undo_poly = QPushButton("撤销描迹\nCtrl+Shift+Z")
        self.btn_undo_poly.clicked.connect(self.undo_poly)
        self.btn_clear = QPushButton("清空全部")
        self.btn_clear.clicked.connect(self.clear_all)
        row3.addWidget(self.btn_undo_poly)
        row3.addWidget(self.btn_clear)
        rl.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self.prev_img)
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self.next_img)
        row4.addWidget(self.btn_prev)
        row4.addWidget(self.btn_next)
        rl.addLayout(row4)

        self.btn_save = QPushButton("保存\nCtrl+S")
        self.btn_save.clicked.connect(self.save)
        rl.addWidget(self.btn_save)

        self.btn_fit = QPushButton("适应窗口 (F)")
        self.btn_fit.clicked.connect(self.canvas.fit)
        rl.addWidget(self.btn_fit)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        rl.addWidget(self.lbl_status)

        # ---- 主布局 ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())

        # ---- 快捷键 ----
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_point)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.undo_poly)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence("Left"), self, activated=self.prev_img)
        QShortcut(QKeySequence("Right"), self, activated=self.next_img)
        QShortcut(QKeySequence("F"), self, activated=self.canvas.fit)
        QShortcut(QKeySequence("Space"), self, activated=self._space_pressed)

        self._space_down = False

        if folder:
            self.load_folder(folder)

    # ---------- 文件夹 ----------
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            self.load_folder(d)

    def load_folder(self, folder: str):
        self.folder = folder
        self.images = []
        self.list_imgs.blockSignals(True)
        self.list_imgs.clear()
        try:
            for name in sorted(os.listdir(folder)):
                p = os.path.join(folder, name)
                if name.lower().endswith(IMG_EXTS) and os.path.isfile(p):
                    self.images.append(p)
                    self.list_imgs.addItem(QListWidgetItem(name))
        except OSError as e:
            QMessageBox.warning(self, "错误", f"无法读取文件夹:\n{e}")
        self.list_imgs.blockSignals(False)
        self.lbl_folder.setText(folder)
        if self.images:
            self.list_imgs.setCurrentRow(0)
        else:
            self.index = -1
            self.traces = []
            self.current_poly = None
            self._refresh()

    # ---------- 图片切换 ----------
    def on_img_selected(self, row):
        if row < 0 or row >= len(self.images):
            return
        if row == self.index:
            return
        # 自动保存当前(无论是否有改动)
        if self.index >= 0:
            self.save()
        self.index = row
        path = self.images[row]
        self.canvas.set_image(path)
        self.load_json_for_current()
        self.dirty = False
        self._refresh()
        self.statusBar().showMessage(f"{row + 1}/{len(self.images)}  {os.path.basename(path)}")
        QTimer.singleShot(0, self.canvas.fit)

    def prev_img(self):
        if self.index > 0:
            self.list_imgs.setCurrentRow(self.index - 1)

    def next_img(self):
        if self.index < len(self.images) - 1:
            self.list_imgs.setCurrentRow(self.index + 1)

    # ---------- 戳点 ----------
    def on_point(self, pos: QPointF):
        if self.current_poly is None:
            self.current_poly = {
                "label": self.combo_label.currentText().strip() or "trace",
                "points": [],
            }
        x, y = int(round(pos.x())), int(round(pos.y()))
        self.current_poly["points"].append([x, y])
        self.dirty = True
        self._refresh()

    def finish_poly(self):
        if self.current_poly and len(self.current_poly["points"]) >= 2:
            self.traces.append(self.current_poly)
            self.current_poly = None
            self.dirty = True
            self._refresh()
        elif self.current_poly:
            # 点太少, 丢弃
            self.current_poly = None
            self._refresh()

    def undo_point(self):
        if self.current_poly and self.current_poly["points"]:
            self.current_poly["points"].pop()
            if not self.current_poly["points"]:
                self.current_poly = None
            self.dirty = True
            self._refresh()
        elif self.traces:
            # 当前没在画, 撤销最后一条描迹
            self.current_poly = self.traces.pop()
            self.dirty = True
            self._refresh()

    def undo_poly(self):
        if self.current_poly:
            self.current_poly = None
            self.dirty = True
            self._refresh()
        elif self.traces:
            self.traces.pop()
            self.dirty = True
            self._refresh()

    def clear_all(self):
        if not self.traces and not self.current_poly:
            return
        if QMessageBox.question(self, "清空", "确定清空当前图片的所有描迹?") == QMessageBox.Yes:
            self.traces = []
            self.current_poly = None
            self.dirty = True
            self._refresh()

    # ---------- JSON ----------
    def _json_path(self) -> str:
        if self.index < 0 or not self.images:
            return ""
        stem = os.path.splitext(os.path.basename(self.images[self.index]))[0]
        return os.path.join(self.folder, stem + ".json")

    def load_json_for_current(self):
        self.traces = []
        self.current_poly = None
        jp = self._json_path()
        if jp and os.path.exists(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.traces = data.get("polylines", [])
            except (OSError, json.JSONDecodeError) as e:
                QMessageBox.warning(self, "加载", f"读取已有 JSON 失败:\n{e}")

    def save(self):
        if self.index < 0 or not self.images:
            return
        # 完成当前正在画的
        if self.current_poly and len(self.current_poly["points"]) >= 2:
            self.traces.append(self.current_poly)
            self.current_poly = None
        jp = self._json_path()
        if not jp:
            return
        total = sum(len(p.get("points", [])) for p in self.traces)
        data = {"polylines": self.traces, "total_points": total}
        try:
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.dirty = False
            self.statusBar().showMessage(f"已保存: {jp}", 5000)
        except OSError as e:
            QMessageBox.warning(self, "保存", f"保存失败:\n{e}")
        self._refresh()

    # ---------- 刷新 ----------
    def _refresh(self):
        self.canvas.draw_traces(self.traces, self.current_poly)
        n_fin = len(self.traces)
        n_cur = len(self.current_poly["points"]) if self.current_poly else 0
        total = sum(len(p.get("points", [])) for p in self.traces) + n_cur
        cur_label = self.current_poly["label"] if self.current_poly else "-"
        self.lbl_status.setText(
            f"已完成描迹: {n_fin}\n"
            f"当前描迹: {cur_label}  ({n_cur} 点)\n"
            f"总点数: {total}\n"
            f"{'(未保存)' if self.dirty else '(已保存)'}")
        title = "电离图描迹标注"
        if self.index >= 0 and self.images:
            title = f"{os.path.basename(self.images[self.index])} - {title}"
        if self.dirty:
            title = "* " + title
        self.setWindowTitle(title)

    # ---------- 空格平移 ----------
    def _space_pressed(self):
        self._space_down = True
        self.canvas.setDragMode(QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space_down = False
            self.canvas.setDragMode(QGraphicsView.NoDrag)
        super().keyReleaseEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        # 窗口首次显示后 viewport 才有正确尺寸, 此时再 fit 一次
        if self.index >= 0:
            QTimer.singleShot(0, self.canvas.fit)

    def closeEvent(self, e):
        if self.dirty:
            r = QMessageBox.question(self, "退出", "有未保存的改动, 保存后退出?")
            if r == QMessageBox.Yes:
                self.save()
            elif r == QMessageBox.Cancel:
                e.ignore()
                return
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    folder = sys.argv[1] if len(sys.argv) > 1 else ""
    w = MainWindow(folder)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
