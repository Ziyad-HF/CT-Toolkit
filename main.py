import sys
import os
import numpy as np
import pydicom
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget, 
                             QSlider, QComboBox, QLabel, QPushButton, QFileDialog, QMessageBox,
                             QGroupBox, QSpinBox, QRadioButton, QFormLayout, QMenu)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.ndimage import zoom, gaussian_filter, median_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal.windows import hann, hamming
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import QTimer
import threading
from skimage.util import img_as_float
from scipy.signal import convolve2d
from PyQt5.QtGui import QCursor
import matplotlib.pyplot as plt
import imageio

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced CT Reconstruction Tool")
        
        # Initialize variables
        self.image_stack = None
        self.interpolated_stack = None
        self.averaged_stack = None
        self.current_stack = None
        self.original_thickness = 1.0
        self.current_thickness = self.original_thickness
        self.current_slice_idx = 0
        self.metadata = {}
        self.filters = ["Ramp", "Hann", "Hamming", "None"]
        self.window_presets = {
            "Brain": (80, 40),
            "Abdomen": (400, 40),
            "Bone": (1500, 300),
            "Spine": (1000, 250)
        }
        self.averaging_filters = ["Mean", "Gaussian", "Median"]
        self.progress_dialog = None
        self.initUI()

    def initUI(self):
        # Main layout: Horizontal split
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Control panel (1/4)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        main_layout.addWidget(control_widget, stretch=1)
        
        # Right: Image display (3/4)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)
        self.canvas.setToolTip("Right-click for options")
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setToolTip("Toolbar for plot interactions: save, zoom, pan, etc.")
        image_layout.addWidget(self.toolbar)
        image_layout.addWidget(self.canvas)
        main_layout.addWidget(image_widget, stretch=3)
        
        # Enable context menu for canvas
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)
    
        # Folder selection group
        folder_group = QGroupBox("DICOM Folder")
        folder_layout = QVBoxLayout(folder_group)
        self.folder_button = QPushButton("Select DICOM Folder")
        self.folder_button.setToolTip("Choose a folder containing DICOM (.dcm) files")
        self.folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_button)
        control_layout.addWidget(folder_group)
        
        # Metadata display
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        self.metadata_label = QLabel("No dataset loaded")
        self.metadata_label.setToolTip("Displays Patient ID, slice count, resolution, and thickness")
        metadata_layout.addWidget(self.metadata_label)
        control_layout.addWidget(metadata_group)
        
        # Filter selection group
        filter_group = QGroupBox("Reconstruction Filter")
        filter_layout = QFormLayout(filter_group)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.filters)
        self.filter_combo.setToolTip("Select filter type (Ramp: sharp, Hann/Hamming: smooth)")
        self.filter_combo.currentTextChanged.connect(self.update_image)
        filter_layout.addRow(QLabel("Filter Type:"), self.filter_combo)
        
        self.domain_freq_radio = QRadioButton("Frequency Domain")
        self.domain_spatial_radio = QRadioButton("Spatial Domain")
        self.domain_freq_radio.setChecked(True)
        self.domain_freq_radio.setToolTip("Apply filter in frequency domain (simulates FBP)")
        self.domain_spatial_radio.setToolTip("Apply filter in spatial domain (post-processing)")
        self.domain_freq_radio.toggled.connect(self.update_image)
        filter_layout.addRow(QLabel("Domain:"), self.domain_freq_radio)
        filter_layout.addRow(QLabel(""), self.domain_spatial_radio)
        control_layout.addWidget(filter_group)
        
        # Slice control group
        slice_group = QGroupBox("Slice Control")
        slice_layout = QFormLayout(slice_group)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setToolTip("Browse through slices in the 3D volume")
        self.slice_slider.valueChanged.connect(self.update_image)
        self.slice_idx_label = QLabel("Slice: 0 / 0")
        self.slice_idx_label.setMinimumWidth(100)
        self.slice_idx_label.setToolTip("Current slice index and total slices")
        slice_layout.addRow(QLabel("Slice Index:"), self.slice_slider)
        slice_layout.addRow(QLabel(""), self.slice_idx_label)
        
        self.thickness_spinbox = QSpinBox()
        self.thickness_spinbox.setRange(1, 20)
        self.thickness_spinbox.setValue(int(self.original_thickness))
        self.thickness_spinbox.setToolTip("Set slice thickness in mm")
        self.thickness_spinbox.valueChanged.connect(self.update_image)
        slice_layout.addRow(QLabel("Slice Thickness (mm):"), self.thickness_spinbox)
        
        self.avg_filter_combo = QComboBox()
        self.avg_filter_combo.addItems(self.averaging_filters)
        self.avg_filter_combo.setToolTip("Select averaging filter for thicker slices")
        self.avg_filter_combo.currentTextChanged.connect(self.update_image)
        self.avg_filter_combo.setEnabled(False)
        slice_layout.addRow(QLabel("Averaging Filter:"), self.avg_filter_combo)
        control_layout.addWidget(slice_group)
        
        # Windowing group
        window_group = QGroupBox("Windowing")
        window_layout = QFormLayout(window_group)
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(-1000, 1000)
        self.wl_slider.setValue(40)
        self.wl_slider.setToolTip("Window level in HU (controls brightness)")
        self.wl_slider.valueChanged.connect(self.update_image)
        window_layout.addRow(QLabel("Window Level (HU):"), self.wl_slider)
        
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(1, 3000)
        self.ww_slider.setValue(400)
        self.ww_slider.setToolTip("Window width in HU (controls contrast)")
        self.ww_slider.valueChanged.connect(self.update_image)
        window_layout.addRow(QLabel("Window Width (HU):"), self.ww_slider)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Manual"] + list(self.window_presets.keys()))
        self.preset_combo.setToolTip("Select windowing preset or adjust manually")
        self.preset_combo.currentTextChanged.connect(self.apply_window_preset)
        window_layout.addRow(QLabel("Preset:"), self.preset_combo)
        control_layout.addWidget(window_group)
        
        # Reset button
        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Reset all parameters to default")
        reset_button.clicked.connect(self.reset_parameters)
        control_layout.addWidget(reset_button)
        
        control_layout.addStretch()

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder_path:
            try:
                self.image_stack, self.metadata = self.load_dicom_stack(folder_path)
                self.current_stack = self.image_stack
                self.original_thickness = self.metadata.get("SliceThickness", 1.0)
                self.current_thickness = self.original_thickness
                self.thickness_spinbox.setValue(int(self.original_thickness))
                self.current_slice_idx = self.image_stack.shape[0] // 2
                self.slice_slider.setMaximum(self.image_stack.shape[0] - 1)
                self.slice_slider.setValue(self.current_slice_idx)
                self.slice_slider.setEnabled(True)
                self.interpolated_stack = None
                self.averaged_stack = None
                self.update_metadata_display()
                self.update_image()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM files: {str(e)}")
                self.image_stack = None
                self.current_stack = None
                self.slice_slider.setEnabled(False)
                self.ax.clear()
                self.canvas.draw()

    def interpolate_volume(self, new_thickness):
        """Interpolate the volume to create thinner slices."""
        ratio = self.original_thickness / new_thickness
        num_slices = int(np.ceil(self.image_stack.shape[0] * ratio))  # Increase slice count
        zoom_factor = (num_slices / self.image_stack.shape[0], 1, 1)
        
        # Initialize progress dialog
        self.progress_dialog = QProgressDialog("Interpolating slices...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        # Perform interpolation in a separate thread
        def interpolate():
            interpolated = zoom(self.image_stack, zoom_factor, order=1)
            self.interpolated_stack = interpolated
            self.current_stack = interpolated
            QTimer.singleShot(0, lambda: self.progress_dialog.setValue(100))
            QTimer.singleShot(0, self.update_after_interpolation)
        
        threading.Thread(target=interpolate, daemon=True).start()
        
    def average_volume(self, new_thickness):
        """Average slices to create thicker slices."""
        ratio = new_thickness / self.original_thickness
        num_slices = int(np.round(self.image_stack.shape[0] / ratio))
        if num_slices < 1:
            num_slices = 1
        
        # Initialize progress dialog
        self.progress_dialog = QProgressDialog("Averaging slices...", "Cancel", 0, num_slices, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        # Perform averaging
        averaged_stack = np.zeros((num_slices, self.image_stack.shape[1], self.image_stack.shape[2]), dtype=np.float32)
        slices_per_group = max(1, int(np.ceil(self.image_stack.shape[0] / num_slices)))
        
        for i in range(num_slices):
            start_idx = i * slices_per_group
            end_idx = min(start_idx + slices_per_group, self.image_stack.shape[0])
            if start_idx < end_idx:
                slice_group = self.image_stack[start_idx:end_idx]
                avg_filter = self.avg_filter_combo.currentText()
                if avg_filter == "Mean":
                    averaged_stack[i] = np.mean(slice_group, axis=0)
                elif avg_filter == "Gaussian":
                    averaged_stack[i] = gaussian_filter(np.mean(slice_group, axis=0), sigma=ratio / 2)
                elif avg_filter == "Median":
                    averaged_stack[i] = median_filter(np.mean(slice_group, axis=0), size=3)
            self.progress_dialog.setValue(i + 1)
        
        self.averaged_stack = averaged_stack
        self.current_stack = averaged_stack
        self.progress_dialog.setValue(num_slices)
        self.update_after_interpolation()

    def update_after_interpolation(self):
        """Update GUI after interpolation or averaging."""
        if self.current_stack is not None:
            # Adjust slice index to maintain approximate spatial position
            thickness_ratio = self.current_thickness / self.original_thickness
            self.current_slice_idx = min(
                int(self.current_slice_idx * thickness_ratio),
                self.current_stack.shape[0] - 1
            )
            self.slice_slider.setMaximum(self.current_stack.shape[0] - 1)
            self.slice_slider.setValue(self.current_slice_idx)
            self.slice_idx_label.setText(f"Slice: {self.current_slice_idx} / {self.current_stack.shape[0] - 1}")
            self.update_image()
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

    def load_dicom_stack(self, folder_path):
        """Load DICOM files into a 3D NumPy array and extract metadata."""
        dicom_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]
        if not dicom_files:
            raise ValueError("No DICOM files found in the folder.")
        
        # Read one file to get dimensions and metadata
        first_slice = pydicom.dcmread(os.path.join(folder_path, dicom_files[0]))
        height, width = first_slice.pixel_array.shape
        depth = len(dicom_files)
        
        # Initialize 3D array
        stack = np.zeros((depth, height, width), dtype=np.float32)
        
        # Load slices, sorted by InstanceNumber
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(os.path.join(folder_path, x)).InstanceNumber)
        for i, fname in enumerate(dicom_files_sorted):
            ds = pydicom.dcmread(os.path.join(folder_path, fname))
            stack[i] = ds.pixel_array
        
        # Extract metadata
        metadata = {
            "PatientID": getattr(first_slice, "PatientID", "Unknown"),
            "SliceCount": depth,
            "Resolution": f"{width}x{height}",
            "SliceThickness": getattr(first_slice, "SliceThickness", 1.0)
        }
        
        return stack, metadata

    def update_metadata_display(self):
        """Update the metadata display label."""
        text = (f"Patient ID: {self.metadata.get('PatientID', 'Unknown')}\n"
                f"Slices: {self.metadata.get('SliceCount', 0)}\n"
                f"Resolution: {self.metadata.get('Resolution', 'Unknown')}\n"
                f"Slice Thickness: {self.metadata.get('SliceThickness', 1.0)} mm")
        self.metadata_label.setText(text)

    def apply_reconstruction_filter(self, img, filter_name="Ramp", domain="frequency"):
        """Apply Ramp, Hann, or Hamming reconstruction filter in frequency or spatial domain."""
        if filter_name == "None":
            return img  # No filter applied

        img = img_as_float(img)  # Normalize image to float [0,1]

        if domain == "frequency":
            # FFT
            img_fft = fftshift(fft2(img))

            ny, nx = img.shape
            u = np.fft.fftfreq(nx).reshape(1, -1)
            v = np.fft.fftfreq(ny).reshape(-1, 1)
            freq = np.sqrt(u**2 + v**2)
            freq /= freq.max()  # Normalize

            # Construct filters
            if filter_name == "Ramp":
                filt = freq
            elif filter_name == "Hann":
                filt = freq * (0.5 * (1 + np.cos(np.pi * freq)))
            elif filter_name == "Hamming":
                filt = freq * (0.54 + 0.46 * np.cos(np.pi * freq))
            else:
                filt = np.ones_like(freq)

            filt = np.clip(filt, 0, 1)

            # Apply filter and reconstruct
            filtered_fft = img_fft * filt
            filtered_img = np.real(ifft2(ifftshift(filtered_fft)))

        else:
            # Spatial domain
            if filter_name == "Ramp":
                # Use Laplacian as a rough high-pass Ramp approximation
                kernel = np.array([[0, -1, 0],
                                [-1,  4, -1],
                                [0, -1, 0]])
            elif filter_name == "Hann":
                w = hann(5)
                kernel = np.outer(w, w)
                kernel /= kernel.sum()
            elif filter_name == "Hamming":
                w = hamming(5)
                kernel = np.outer(w, w)
                kernel /= kernel.sum()
            else:
                kernel = np.ones((3, 3)) / 9

            filtered_img = convolve2d(img, kernel, mode='same', boundary='symm')

        return filtered_img


    def adjust_slice_thickness(self, slice_idx, new_thickness):
        """Adjust slice thickness by selecting from precomputed volumes."""
        if new_thickness != self.current_thickness:
            self.current_thickness = new_thickness
            if new_thickness == self.original_thickness:
                self.current_stack = self.image_stack
                self.avg_filter_combo.setEnabled(False)
                self.interpolated_stack = None
                self.averaged_stack = None
                self.slice_slider.setMaximum(self.current_stack.shape[0] - 1)
                self.current_slice_idx = min(slice_idx, self.current_stack.shape[0] - 1)
                self.slice_idx_label.setText(f"Slice: {self.current_slice_idx} / {self.current_stack.shape[0] - 1}")
            elif new_thickness < self.original_thickness:
                self.avg_filter_combo.setEnabled(False)
                self.interpolate_volume(new_thickness)
                return self.current_stack[self.current_slice_idx] if self.current_stack is not None else self.image_stack[slice_idx]
            else:
                self.avg_filter_combo.setEnabled(True)
                self.average_volume(new_thickness)
                return self.current_stack[self.current_slice_idx] if self.current_stack is not None else self.image_stack[slice_idx]
        
        self.avg_filter_combo.setEnabled(new_thickness > self.original_thickness)
        return self.current_stack[slice_idx]

    def apply_window_preset(self):
        """Apply windowing preset or revert to manual."""
        preset = self.preset_combo.currentText()
        if preset != "Manual":
            ww, wl = self.window_presets[preset]
            self.ww_slider.setValue(ww)
            self.wl_slider.setValue(wl)
        self.update_image()

    def process_image(self, img, slice_idx):
        """Apply reconstruction filter, slice thickness, and windowing."""
        # Apply slice thickness
        new_thickness = self.thickness_spinbox.value()
        processed_img = self.adjust_slice_thickness(slice_idx, new_thickness)
        
        # Apply reconstruction filter
        filter_name = self.filter_combo.currentText()
        domain = "frequency" if self.domain_freq_radio.isChecked() else "spatial"
        processed_img = self.apply_reconstruction_filter(processed_img, filter_name, domain)
        
        # Apply windowing
        wl = self.wl_slider.value()
        ww = self.ww_slider.value()
        min_hu = wl - ww / 2
        max_hu = wl + ww / 2
        windowed_img = np.clip((processed_img - min_hu) / (max_hu - min_hu) * 255, 0, 255)
        return windowed_img.astype(np.uint8)

    def reset_parameters(self):
        """Reset all parameters to default."""
        self.current_stack = self.image_stack
        self.interpolated_stack = None
        self.averaged_stack = None
        self.current_thickness = self.original_thickness
        self.thickness_spinbox.setValue(int(self.original_thickness))
        self.slice_slider.setValue(self.image_stack.shape[0] // 2 if self.image_stack is not None else 0)
        self.slice_slider.setMaximum(self.image_stack.shape[0] - 1 if self.image_stack is not None else 0)
        self.filter_combo.setCurrentText("Hann")
        self.domain_freq_radio.setChecked(True)
        self.avg_filter_combo.setCurrentText("Mean")
        self.wl_slider.setValue(40)
        self.ww_slider.setValue(400)
        self.preset_combo.setCurrentText("Manual")
        self.update_image()

    def update_image(self):
        """Update the displayed image based on current parameters."""
        if self.current_stack is None or self.progress_dialog:
            return
        self.current_slice_idx = self.slice_slider.value()
        self.slice_idx_label.setText(f"Slice: {self.current_slice_idx} / {self.current_stack.shape[0] - 1}")
        processed_image = self.process_image(self.current_stack[self.current_slice_idx], self.current_slice_idx)
        self.ax.clear()
        self.ax.imshow(processed_image, cmap='gray')
        self.ax.axis('off')
        self.canvas.draw()

    def show_context_menu(self, point):
        """Show context menu on right-click of the canvas."""
        if self.current_stack is None:
            return
        menu = QMenu(self)
        export_slices_action = menu.addAction("Export All Slices as Image")
        export_slices_action.triggered.connect(self.export_all_slices)
        menu.exec_(QCursor.pos())

    def export_all_slices(self):
        """Export all slices as a single tiled image."""
        if self.current_stack is None:
            QMessageBox.warning(self, "Warning", "No volume loaded to export.")
            return
        
        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Slices Image", "", "PNG Files (*.png)")
        if not save_path:
            return
        
        # Determine grid size
        num_slices = self.current_stack.shape[0]
        grid_cols = int(np.ceil(np.sqrt(num_slices)))
        grid_rows = int(np.ceil(num_slices / grid_cols))
        
        # Create figure for tiling
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
        axes = axes.ravel() if num_slices > 1 else [axes]
        
        # Process and display each slice
        for i in range(num_slices):
            processed_slice = self.process_image(self.current_stack[i], i)
            axes[i].imshow(processed_slice, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Slice {i}", fontsize=8)
        
        # Hide unused subplots
        for i in range(num_slices, len(axes)):
            axes[i].axis('off')
        
        # Save and close
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        QMessageBox.information(self, "Success", f"Slices exported to {save_path}")


def laod_style_sheet(file_path):
    """Load and apply the stylesheet for the application."""
    with open(file_path, "r") as f:
        style = f.read()
    return style

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(laod_style_sheet("style.qss"))  
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())