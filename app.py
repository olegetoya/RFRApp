import sys
import csv
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inference.detector import RFRDetector


DEFAULT_CONFIG = "configs/rfr_models.json"
DEFAULT_MODEL = "ResUNet_RFR"
DEFAULT_OUTPUT_DIR = "outputs/gui_result"

FIELDNAMES = [
    "frame_idx",
    "frame_name",
    "object_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "area",
    "mask_path",
    "overlay_path",
    "model_name",
]


def save_results_csv(results, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(FIELDNAMES)

    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


class InferenceWorker(QThread):
    log_message = Signal(str)
    progress_changed = Signal(int, int, str, int)
    finished_success = Signal(list, str, bool)
    finished_error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        checkpoint_path: str,
        config_path: str,
        output_dir: str,
        model_name: str,
        device: str | None,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def should_stop(self):
        return self._stop_requested

    def _on_progress(self, current, total, frame_name, objects_count):
        self.progress_changed.emit(current, total, frame_name, objects_count)

    def run(self):
        try:
            self.log_message.emit("Создание детектора...")

            detector = RFRDetector(
                config_path=self.config_path,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device if self.device else None,
            )

            self.log_message.emit("Запуск инференса...")

            results = detector.predict_folder(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                should_stop=self.should_stop,
                progress_callback=self._on_progress,
            )

            output_csv = Path(self.output_dir) / "results.csv"
            save_results_csv(results, output_csv)

            if self._stop_requested:
                self.log_message.emit("Обработка остановлена пользователем.")
            else:
                self.log_message.emit("Инференс завершён.")

            self.log_message.emit(f"CSV сохранён: {output_csv}")
            self.finished_success.emit(results, self.output_dir, self._stop_requested)

        except Exception as exc:
            self.finished_error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RFR Satellite Target Detector")
        self.resize(1200, 760)

        self.worker = None
        self.results = []
        self.overlay_paths = []
        self.current_overlay_index = 0

        self._build_ui()

    def _stop_inference(self):
        if self.worker is not None and self.worker.isRunning():
            self._append_log("Запрошена остановка. Дождитесь завершения текущего кадра...")
            self.stop_btn.setEnabled(False)
            self.worker.request_stop()

    def _on_progress_changed(self, current, total, frame_name, objects_count):
        self.progress.setRange(0, total)
        self.progress.setValue(current)

        self._append_log(
            f"{current}/{total} {frame_name}: objects={objects_count}"
        )

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        settings_group = QGroupBox("Настройки запуска")
        settings_layout = QGridLayout(settings_group)

        self.input_dir_edit = QLineEdit()
        self.checkpoint_edit = QLineEdit("checkpoints/ResUNet_RFR.pth.tar")
        self.config_edit = QLineEdit(DEFAULT_CONFIG)
        self.output_dir_edit = QLineEdit(DEFAULT_OUTPUT_DIR)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "ResUNet_RFR",
            "ISTUDNet_RFR",
            "DNANet_RFR",
            "ALCNet_RFR",
            "ACM_RFR",
        ])
        self.model_combo.setCurrentText(DEFAULT_MODEL)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.device_combo.setCurrentText("auto")

        input_btn = QPushButton("Выбрать...")
        input_btn.clicked.connect(self._choose_input_dir)

        checkpoint_btn = QPushButton("Выбрать...")
        checkpoint_btn.clicked.connect(self._choose_checkpoint)

        config_btn = QPushButton("Выбрать...")
        config_btn.clicked.connect(self._choose_config)

        output_btn = QPushButton("Выбрать...")
        output_btn.clicked.connect(self._choose_output_dir)

        settings_layout.addWidget(QLabel("Папка с кадрами:"), 0, 0)
        settings_layout.addWidget(self.input_dir_edit, 0, 1)
        settings_layout.addWidget(input_btn, 0, 2)

        settings_layout.addWidget(QLabel("Checkpoint:"), 1, 0)
        settings_layout.addWidget(self.checkpoint_edit, 1, 1)
        settings_layout.addWidget(checkpoint_btn, 1, 2)

        settings_layout.addWidget(QLabel("Конфиг моделей:"), 2, 0)
        settings_layout.addWidget(self.config_edit, 2, 1)
        settings_layout.addWidget(config_btn, 2, 2)

        settings_layout.addWidget(QLabel("Папка результата:"), 3, 0)
        settings_layout.addWidget(self.output_dir_edit, 3, 1)
        settings_layout.addWidget(output_btn, 3, 2)

        settings_layout.addWidget(QLabel("Модель:"), 4, 0)
        settings_layout.addWidget(self.model_combo, 4, 1)

        settings_layout.addWidget(QLabel("Устройство:"), 5, 0)
        settings_layout.addWidget(self.device_combo, 5, 1)

        main_layout.addWidget(settings_group)

        buttons_layout = QHBoxLayout()

        self.run_btn = QPushButton("Запустить обработку")
        self.run_btn.clicked.connect(self._run_inference)

        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)

        self.open_output_btn = QPushButton("Открыть папку результата")
        self.open_output_btn.clicked.connect(self._open_output_dir)

        buttons_layout.addWidget(self.run_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.open_output_btn)

        main_layout.addLayout(buttons_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        content_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Лог работы приложения...")

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "frame_idx",
            "frame_name",
            "object_id",
            "x_center",
            "y_center",
            "width",
            "height",
            "area",
            "model_name",
            "overlay_path",
        ])

        left_layout.addWidget(QLabel("Лог:"))
        left_layout.addWidget(self.log_box, 1)
        left_layout.addWidget(QLabel("Найденные объекты:"))
        left_layout.addWidget(self.table, 2)

        right_layout = QVBoxLayout()

        self.preview_label = QLabel("Overlay preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(420, 420)
        self.preview_label.setStyleSheet("border: 1px solid gray; background: #202020; color: white;")

        preview_buttons = QHBoxLayout()

        self.prev_btn = QPushButton("← Предыдущий")
        self.prev_btn.clicked.connect(self._show_prev_overlay)

        self.next_btn = QPushButton("Следующий →")
        self.next_btn.clicked.connect(self._show_next_overlay)

        preview_buttons.addWidget(self.prev_btn)
        preview_buttons.addWidget(self.next_btn)

        self.preview_info = QLabel("Нет изображений для просмотра")
        self.preview_info.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(QLabel("Просмотр результата:"))
        right_layout.addWidget(self.preview_label, 1)
        right_layout.addLayout(preview_buttons)
        right_layout.addWidget(self.preview_info)

        content_layout.addLayout(left_layout, 2)
        content_layout.addLayout(right_layout, 1)

        main_layout.addLayout(content_layout, 1)

        self.setCentralWidget(central)

    def _choose_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Выбрать папку с кадрами")
        if directory:
            self.input_dir_edit.setText(directory)

    def _choose_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать checkpoint",
            "",
            "PyTorch checkpoints (*.pth *.pth.tar *.pt *.ckpt);;All files (*.*)",
        )
        if file_path:
            self.checkpoint_edit.setText(file_path)

    def _choose_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать config",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if file_path:
            self.config_edit.setText(file_path)

    def _choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Выбрать папку результата")
        if directory:
            self.output_dir_edit.setText(directory)

    def _append_log(self, message: str):
        self.log_box.appendPlainText(message)

    def _validate_inputs(self):
        input_dir = self.input_dir_edit.text().strip()
        checkpoint = self.checkpoint_edit.text().strip()
        config = self.config_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not input_dir:
            QMessageBox.warning(self, "Ошибка", "Выбери папку с кадрами.")
            return None

        if not Path(input_dir).exists():
            QMessageBox.warning(self, "Ошибка", f"Папка с кадрами не найдена:\n{input_dir}")
            return None

        if not checkpoint:
            QMessageBox.warning(self, "Ошибка", "Укажи checkpoint модели.")
            return None

        if not Path(checkpoint).exists():
            QMessageBox.warning(self, "Ошибка", f"Checkpoint не найден:\n{checkpoint}")
            return None

        if not config:
            QMessageBox.warning(self, "Ошибка", "Укажи JSON-конфиг.")
            return None

        if not Path(config).exists():
            QMessageBox.warning(self, "Ошибка", f"Конфиг не найден:\n{config}")
            return None

        if not output_dir:
            QMessageBox.warning(self, "Ошибка", "Укажи папку результата.")
            return None

        return input_dir, checkpoint, config, output_dir

    def _run_inference(self):
        values = self._validate_inputs()
        if values is None:
            return

        input_dir, checkpoint, config, output_dir = values

        model_name = self.model_combo.currentText()
        device_text = self.device_combo.currentText()
        device = None if device_text == "auto" else device_text

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setRange(0, 0)
        self.log_box.clear()
        self.table.setRowCount(0)
        self.overlay_paths = []
        self.current_overlay_index = 0
        self.preview_label.setText("Обработка...")
        self.preview_info.setText("")

        self._append_log("Запуск обработки...")
        self._append_log(f"input_dir: {input_dir}")
        self._append_log(f"checkpoint: {checkpoint}")
        self._append_log(f"config: {config}")
        self._append_log(f"output_dir: {output_dir}")
        self._append_log(f"model_name: {model_name}")
        self._append_log(f"device: {device_text}")

        self.worker = InferenceWorker(
            input_dir=input_dir,
            checkpoint_path=checkpoint,
            config_path=config,
            output_dir=output_dir,
            model_name=model_name,
            device=device,
        )

        self.worker.log_message.connect(self._append_log)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.finished_success.connect(self._on_inference_success)
        self.worker.finished_error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_success(self, results: list, output_dir: str, was_stopped: bool):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

        self.results = results

        self._append_log(f"Готово. Найдено объектов: {len(results)}")
        self._append_log(f"Папка результата: {output_dir}")

        self._fill_table(results)
        self._collect_overlays(results, output_dir)
        self._show_current_overlay()

        if was_stopped:
            QMessageBox.information(
                self,
                "Остановлено",
                f"Обработка остановлена.\nСохранены частичные результаты.\nНайдено объектов: {len(results)}"
            )
        else:
            QMessageBox.information(
                self,
                "Готово",
                f"Обработка завершена.\nНайдено объектов: {len(results)}"
            )

    def _on_inference_error(self, error_text: str):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self._append_log("Ошибка:")
        self._append_log(error_text)

        QMessageBox.critical(self, "Ошибка", error_text)

    def _fill_table(self, results: list):
        self.table.setRowCount(len(results))

        columns = [
            "frame_idx",
            "frame_name",
            "object_id",
            "x_center",
            "y_center",
            "width",
            "height",
            "area",
            "model_name",
            "overlay_path",
        ]

        for row_idx, row in enumerate(results):
            for col_idx, key in enumerate(columns):
                value = row.get(key, "")
                item = QTableWidgetItem(str(value))
                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()

    def _collect_overlays(self, results: list, output_dir: str):
        paths = []

        for row in results:
            overlay = row.get("overlay_path")
            if overlay and overlay not in paths:
                paths.append(overlay)

        if not paths:
            overlay_dir = Path(output_dir) / "overlays"
            if overlay_dir.exists():
                paths = [str(path) for path in sorted(overlay_dir.glob("*_overlay.png"))]

        self.overlay_paths = paths
        self.current_overlay_index = 0

    def _show_current_overlay(self):
        if not self.overlay_paths:
            self.preview_label.setText("Overlay не найден")
            self.preview_info.setText("Нет изображений для просмотра")
            return

        path = self.overlay_paths[self.current_overlay_index]
        pixmap = QPixmap(path)

        if pixmap.isNull():
            self.preview_label.setText("Не удалось открыть overlay")
            self.preview_info.setText(path)
            return

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview_label.setPixmap(scaled)
        self.preview_info.setText(
            f"{self.current_overlay_index + 1}/{len(self.overlay_paths)}: {Path(path).name}"
        )

    def _show_prev_overlay(self):
        if not self.overlay_paths:
            return

        self.current_overlay_index -= 1
        if self.current_overlay_index < 0:
            self.current_overlay_index = len(self.overlay_paths) - 1

        self._show_current_overlay()

    def _show_next_overlay(self):
        if not self.overlay_paths:
            return

        self.current_overlay_index += 1
        if self.current_overlay_index >= len(self.overlay_paths):
            self.current_overlay_index = 0

        self._show_current_overlay()

    def _open_output_dir(self):
        output_dir = self.output_dir_edit.text().strip()

        if not output_dir:
            QMessageBox.warning(self, "Ошибка", "Папка результата не указана.")
            return

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Windows-specific open
        import os
        os.startfile(str(path.resolve()))

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.overlay_paths:
            self._show_current_overlay()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()