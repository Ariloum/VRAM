# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import pynvml
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import os
import csv
import platform
import sv_ttk
import queue
import xml.etree.ElementTree as ET

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates # For date formatting if needed

# --- Configuration ---
UPDATE_INTERVAL_SECONDS = 1
GPU_DEVICE_INDEX = 0
CSV_FILENAME = 'gpu_stats_log.csv'
DATA_WINDOW_HOURS = 0.5
PERIODIC_CLEANUP_INTERVAL_HOURS = 0.1

# --- Global Flags & Theme ---
DARK_BACKGROUND = '#2e2e2e'
LIGHT_TEXT = '#f0f0f0'
VRAM_COLOR = '#66b3ff'
CUDA_COLOR = '#ffa500'
GRID_COLOR = '#555555'

# --- Platform Specific DPI Awareness (Windows) ---
if platform.system() == "Windows":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        print("Process DPI Awareness set to Per Monitor V2.")
    except AttributeError:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            print("Process DPI Awareness set (fallback).")
        except Exception as e: print(f"Could not set DPI awareness: {e}")
    except Exception as e: print(f"Could not set DPI awareness: {e}")
    
    # Set app ID for Windows 7 and above (helps with taskbar icon)
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VRAM.Monitor.1")
        print("Set explicit AppUserModelID for Windows taskbar.")
    except Exception as e: print(f"Could not set AppUserModelID: {e}")

# --- Matplotlib Font settings ---
try:
    plt.rcParams.update({
        'font.family': 'Segoe UI', 'font.size': 10, 'text.color': LIGHT_TEXT,
        'axes.labelcolor': LIGHT_TEXT, 'xtick.color': LIGHT_TEXT,
        'ytick.color': LIGHT_TEXT, 'axes.edgecolor': LIGHT_TEXT
    })
    print("Matplotlib default font set to Segoe UI.")
except Exception as e: print(f"Could not set Matplotlib default font: {e}")

# --- NVML Initialization & Total VRAM ---
NVML_INITIALIZED = False
device_name = "N/A"; handle = None; total_vram_mb = 0
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_DEVICE_INDEX)
    device_name_raw = pynvml.nvmlDeviceGetName(handle)
    device_name = device_name_raw.decode('utf-8') if isinstance(device_name_raw, bytes) else device_name_raw
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_vram_mb = mem_info.total / (1024**2)
    print(f"Monitoring GPU {GPU_DEVICE_INDEX}: {device_name} - Total VRAM: {total_vram_mb:.2f} MB")
    NVML_INITIALIZED = True
except Exception as e:
    print(f"NVML/GPU Init Error: {e}")
    device_name = "N/A (Error)"


# --- Helper Functions (bytes_to_mb, get_gpu_stats_safe, load_initial_data, _prune_csv_file_core) ---
# (These functions remain largely the same as your previous version)
def bytes_to_mb(bytes_val): return bytes_val / (1024**2)

# --- Settings Functions ---
def load_settings():
    settings = {
        'window': {
            'width': 1800,
            'height': 1400,
            'x': 100,
            'y': 100
        }
    }
    
    try:
        if os.path.exists('settings.xml'):
            tree = ET.parse('settings.xml')
            root = tree.getroot()
            
            window = root.find('window')
            if window is not None:
                width = window.find('width')
                height = window.find('height')
                x = window.find('x')
                y = window.find('y')
                
                if width is not None:
                    settings['window']['width'] = int(width.text)
                if height is not None:
                    settings['window']['height'] = int(height.text)
                if x is not None:
                    settings['window']['x'] = int(x.text)
                if y is not None:
                    settings['window']['y'] = int(y.text)
            
            print("Settings loaded successfully.")
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    return settings

def save_settings(settings):
    try:
        root = ET.Element('settings')
        
        window = ET.SubElement(root, 'window')
        ET.SubElement(window, 'width').text = str(settings['window']['width'])
        ET.SubElement(window, 'height').text = str(settings['window']['height'])
        ET.SubElement(window, 'x').text = str(settings['window']['x'])
        ET.SubElement(window, 'y').text = str(settings['window']['y'])
        
        tree = ET.ElementTree(root)
        tree.write('settings.xml', encoding='utf-8', xml_declaration=True)
        print("Settings saved successfully.")
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_gpu_stats_safe():
    if not NVML_INITIALIZED or handle is None: return 0.0, 0.0, datetime.now()
    try:
        timestamp = datetime.now()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_vram_mb = bytes_to_mb(mem_info.used)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        cuda_util_percent = float(utilization.gpu)
        return used_vram_mb, cuda_util_percent, timestamp
    except pynvml.NVMLError as error:
        print(f"Error fetching GPU stats: {error}")
        return 0.0, 0.0, datetime.now()

def _prune_csv_file_core(filename, hours, context_msg="Pruning"):
    print(f"{context_msg}: Pruning '{filename}' to last {hours} hours...")
    try:
        df_full = pd.read_csv(filename, usecols=['Timestamp', 'VRAM Used (MB)', 'CUDA Util (%)'], parse_dates=['Timestamp'])
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_df = df_full[df_full['Timestamp'] >= cutoff_time].copy()
        recent_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"{context_msg}: CSV pruning complete. Kept {len(recent_df)} records in '{filename}'.")
        return True
    except FileNotFoundError:
        print(f"{context_msg}: CSV file not found, nothing to clean.")
        return False
    except Exception as e:
        print(f"{context_msg}: Error during CSV pruning: {e}")
        return False

def load_initial_data(filename=CSV_FILENAME, hours=DATA_WINDOW_HOURS):
    print(f"Attempting to load data from the last {hours} hours from {filename}...")
    cols = ['Timestamp', 'VRAM Used (MB)', 'CUDA Util (%)']
    try:
        _prune_csv_file_core(filename, hours, "Initial load pruning")
        df = pd.read_csv(filename, parse_dates=['Timestamp'], dtype={'VRAM Used (MB)': float, 'CUDA Util (%)': float})
        cutoff_time = datetime.now() - timedelta(hours=hours) # Ensure conformity
        recent_df = df[df['Timestamp'] >= cutoff_time].copy()
        print(f"Loaded {len(recent_df)} recent records for UI.")
        return recent_df
    except FileNotFoundError:
        print("CSV file not found. Starting with empty data.")
        return pd.DataFrame(columns=cols)
    except Exception as e:
        print(f"Error loading or parsing CSV: {e}. Starting with empty data.")
        return pd.DataFrame(columns=cols)

# --- Tkinter Application Class ---
class GpuMonitorApp:
    def __init__(self, root, initial_df):
        self.root = root
        self.root.title(f"GPU Monitor - {device_name}")
        
        # Set window icon - try multiple approaches
        icon_set = False
        try:
            # Get the absolute path to the icon file
            icon_path = os.path.abspath('app.ico')
            print(f"Looking for icon at: {icon_path}")
            
            if os.path.exists(icon_path):
                # Try the standard approach first
                try:
                    self.root.iconbitmap(icon_path)
                    icon_set = True
                    print("Icon set using iconbitmap.")
                except Exception as e1:
                    print(f"iconbitmap failed: {e1}")
                    
                    # Try alternative approach for some Windows versions
                    try:
                        self.root.iconbitmap(default=icon_path)
                        icon_set = True
                        print("Icon set using iconbitmap(default=).")
                    except Exception as e2:
                        print(f"iconbitmap(default=) failed: {e2}")
                        
                        # Try PhotoImage approach as last resort
                        try:
                            icon = tk.PhotoImage(file=icon_path)
                            self.root.iconphoto(True, icon)
                            icon_set = True
                            print("Icon set using iconphoto.")
                        except Exception as e3:
                            print(f"iconphoto failed: {e3}")
            else:
                print(f"Icon file not found at: {icon_path}")
        except Exception as e:
            print(f"Error setting icon: {e}")
            
        if not icon_set:
            print("WARNING: Could not set application icon.")
        
        # Load settings
        self.settings = load_settings()
        window_width = self.settings['window']['width']
        window_height = self.settings['window']['height']
        window_x = self.settings['window']['x']
        window_y = self.settings['window']['y']
        
        # Set window geometry
        self.TARGET_GEOMETRY = f"{window_width}x{window_height}+{window_x}+{window_y}"
        self.root.geometry(self.TARGET_GEOMETRY)
        sv_ttk.set_theme("dark")

        self.monitoring_active = True
        self.total_vram = total_vram_mb
        self.df = initial_df.copy() # Use a copy

        self.data_queue = queue.Queue()
        self.csv_write_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)

        self.fig = Figure(figsize=(16, 10), dpi=100, facecolor=DARK_BACKGROUND)
        self.ax1 = self.fig.add_subplot(111, facecolor=DARK_BACKGROUND) # CUDA (left)
        self.ax2 = self.ax1.twinx() # VRAM (right)
        self.ax2.set_facecolor(DARK_BACKGROUND)
        self.ax2.set_zorder(self.ax1.get_zorder() + 1)
        self.ax2.patch.set_visible(False)

        # --- Create Line2D objects for plotting ---
        # Plot with initial data (even if empty) to create the line objects
        initial_timestamps = initial_df['Timestamp'] if not initial_df.empty else []
        initial_cuda = initial_df['CUDA Util (%)'] if not initial_df.empty else []
        initial_vram = initial_df['VRAM Used (MB)'] if not initial_df.empty else []

        self.cuda_line, = self.ax1.plot(initial_timestamps, initial_cuda, color=CUDA_COLOR, label='CUDA Util (%)')
        self.vram_line, = self.ax2.plot(initial_timestamps, initial_vram, color=VRAM_COLOR, label='VRAM Used (MB)')

        # --- Static Plot Formatting (done once) ---
        self.ax1.set_xlabel("Time")
        self.ax1.set_ylabel("CUDA Util (%)", color=CUDA_COLOR)
        self.ax2.set_ylabel("VRAM Used (MB)", color=VRAM_COLOR)
        self.ax2.yaxis.set_label_position("right"); self.ax2.yaxis.tick_right()
        self.ax1.tick_params(axis='y', labelcolor=CUDA_COLOR)
        self.ax2.tick_params(axis='y', labelcolor=VRAM_COLOR)
        self.ax1.tick_params(axis='x', rotation=30) # Initial rotation
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) # Format x-axis time

        self.ax1.set_ylim(0, 105)
        if self.total_vram > 0:
            self.ax2.set_ylim(0, self.total_vram * 1.05)
            tick_interval = 1000
            self.ax2.yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
            # Grid on ax1 (bottom layer) but scaled by ax2's ticks
            self.ax1.grid(axis='y', linestyle='--', alpha=0.7, which='major', color=GRID_COLOR)
        else:
            self.ax2.set_ylim(0, 1000)
            self.ax1.grid(False)

        # Set initial x-axis limits based on loaded data or a default small window
        if not self.df.empty:
            min_time, max_time = self.df['Timestamp'].min(), self.df['Timestamp'].max()
            if min_time == max_time : max_time = min_time + timedelta(seconds=60) # Default window if single point
            self.ax1.set_xlim(min_time, max_time)
        else: # Default empty view
             self.ax1.set_xlim(datetime.now(), datetime.now() + timedelta(seconds=60))


        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, facecolor=DARK_BACKGROUND, edgecolor=LIGHT_TEXT, labelcolor=LIGHT_TEXT)
        try: self.fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
        except ValueError: pass # Ignore if it fails sometimes

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=DARK_BACKGROUND)
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20,0))

        self.label_frame = ttk.Frame(self.root, style='Card.TFrame', padding=10)
        self.label_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.vram_text = tk.StringVar(value="VRAM Used: --- MB")
        self.cuda_text = tk.StringVar(value="CUDA Util: --- %")
        label_font = ("Segoe UI", 14)
        ttk.Label(self.label_frame, textvariable=self.cuda_text, font=label_font).pack(side=tk.LEFT, padx=20, pady=10)
        ttk.Label(self.label_frame, textvariable=self.vram_text, font=label_font).pack(side=tk.RIGHT, padx=20, pady=10)

        self.initial_draw_complete = False

        self.data_collector_thread = threading.Thread(target=self._data_collector_loop, daemon=True)
        self.csv_writer_thread = threading.Thread(target=self._csv_writer_loop, daemon=True)
        self.periodic_pruner_thread = threading.Thread(target=self._periodic_pruner_loop, daemon=True)
        self.data_collector_thread.start(); self.csv_writer_thread.start(); self.periodic_pruner_thread.start()

        self.root.after(100, self.process_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(200, self._initial_canvas_draw_if_needed) # For the very first draw

    def _initial_canvas_draw_if_needed(self):
        """Ensures the canvas is drawn at least once after setup."""
        if not self.initial_draw_complete:
            self.canvas.draw() # Initial draw
            self.root.after(50, self._force_redraw) # Then apply the resize trick
            self.initial_draw_complete = True


    def _data_collector_loop(self):
        # (Same as your previous version)
        print("Data collector thread started.")
        while not self.stop_event.is_set():
            used_vram, cuda_util, timestamp = get_gpu_stats_safe()
            if self.monitoring_active:
                self.data_queue.put({'timestamp': timestamp, 'vram': used_vram, 'cuda': cuda_util})
            time_to_sleep = UPDATE_INTERVAL_SECONDS - (datetime.now() - timestamp).total_seconds()
            if time_to_sleep > 0:
                self.stop_event.wait(time_to_sleep)
        print("Data collector thread finished.")

    def _csv_writer_loop(self):
        # (Same as your previous version)
        print("CSV writer thread started.")
        while not self.stop_event.is_set() or not self.csv_write_queue.empty():
            try:
                data_to_write = self.csv_write_queue.get(timeout=0.5)
                if data_to_write is None: self.csv_write_queue.task_done(); break
                filename = CSV_FILENAME; file_exists = os.path.isfile(filename)
                try:
                    with open(filename, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists or os.stat(filename).st_size == 0:
                            writer.writerow(['Timestamp', 'VRAM Used (MB)', 'CUDA Util (%)'])
                        writer.writerow(data_to_write)
                except Exception as e: print(f"Error saving data to CSV in writer thread: {e}")
                self.csv_write_queue.task_done()
            except queue.Empty: continue
        print("CSV writer thread finished.")

    def _periodic_pruner_loop(self):
        # (Same as your previous version)
        print("Periodic CSV pruner thread started.")
        while not self.stop_event.wait(PERIODIC_CLEANUP_INTERVAL_HOURS * 3600):
            if self.stop_event.is_set(): break
            if self.monitoring_active:
                 _prune_csv_file_core(CSV_FILENAME, DATA_WINDOW_HOURS, "Periodic pruning")
        print("Periodic CSV pruner thread finished.")

    def process_ui_queue(self):
        new_data_processed = False
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                new_data_processed = True

                timestamp, used_vram, cuda_util = data['timestamp'], data['vram'], data['cuda']
                self.vram_text.set(f"VRAM Used: {used_vram:.2f} MB")
                self.cuda_text.set(f"CUDA Util: {cuda_util:.1f} %")

                new_data_row = {'Timestamp': timestamp, 'VRAM Used (MB)': used_vram, 'CUDA Util (%)': cuda_util}
                # Use pd.concat for appending to DataFrame
                self.df = pd.concat([self.df, pd.DataFrame([new_data_row])], ignore_index=True)

                cutoff_time = datetime.now() - timedelta(hours=DATA_WINDOW_HOURS)
                self.df = self.df[self.df['Timestamp'] >= cutoff_time]

                csv_row = [timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], f"{used_vram:.2f}", f"{cuda_util:.2f}"]
                if self.monitoring_active: self.csv_write_queue.put(csv_row)

            if new_data_processed and self.initial_draw_complete: # Only update plot if new data and initial draw done
                self.update_plot_elements_optimized()

        except queue.Empty: pass
        except Exception as e: print(f"Error in process_ui_queue: {e}")

        if self.monitoring_active:
            self.root.after(100, self.process_ui_queue)

    def update_plot_elements_optimized(self):
        """Optimized update for Matplotlib plot elements by updating line data."""
        if self.df.empty:
            self.cuda_line.set_data([], [])
            self.vram_line.set_data([], [])
        else:
            timestamps = self.df['Timestamp']
            cuda_utils = self.df['CUDA Util (%)']
            vram_usages = self.df['VRAM Used (MB)']

            self.cuda_line.set_data(timestamps, cuda_utils)
            self.vram_line.set_data(timestamps, vram_usages)

            min_time, max_time = timestamps.min(), timestamps.max()
            if min_time == max_time : max_time = min_time + timedelta(seconds=60) # Avoid zero range

            self.ax1.set_xlim(min_time, max_time)
            # self.ax1.relim() # Recalculate data limits for ax1
            # self.ax2.relim() # Recalculate data limits for ax2
            # self.ax1.autoscale_view(scalex=True, scaley=False) # Autoscale x-axis, keep y-axis fixed
            # self.ax2.autoscale_view(scalex=True, scaley=False) # Autoscale x-axis, keep y-axis fixed

        # Request a redraw when Tkinter is idle
        self.canvas.draw_idle()


    def _force_redraw(self):
        print("Forcing redraw via resize...")
        try:
            current_geometry = self.root.geometry()
            size_part = current_geometry.split('+')[0]; parts = size_part.split('x')
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])
                self.root.geometry(f'{width + 1}x{height + 1}')
                self.root.after(20, lambda: self._restore_size(self.TARGET_GEOMETRY))
            else: self._restore_size(self.TARGET_GEOMETRY)
        except Exception as e:
            print(f"Error during force redraw: {e}"); self._restore_size(self.TARGET_GEOMETRY)

    def _restore_size(self, target_geometry):
        print(f"Restoring geometry to {target_geometry}...")
        try:
            self.root.geometry(target_geometry)
            self.label_frame.update_idletasks()
            self.canvas.draw() # Full draw after resize
            self.canvas_widget.update_idletasks()
            print("Geometry restored.")
        except Exception as e: print(f"Error restoring geometry: {e}")

    def on_closing(self):
        # Prevent multiple calls to on_closing
        if not hasattr(self, '_closing') or not self._closing:
            self._closing = True
        else:
            print("Closing already in progress, ignoring duplicate call")
            return
            
        # Save window position and size
        try:
            geometry = self.root.geometry()
            print(f"Current geometry: {geometry}")
            
            # Parse width and height
            size_part = geometry.split('+')[0]
            width, height = map(int, size_part.split('x'))
            
            # Parse x and y position
            # The format can be either "WIDTHxHEIGHT+X+Y" or sometimes "WIDTHxHEIGHT+-X+-Y"
            parts = geometry.split('+', 2)
            if len(parts) >= 3:  # Format: WIDTHxHEIGHT+X+Y
                x = int(parts[1])
                y = int(parts[2])
            else:  # Try alternative format with negative coordinates
                parts = geometry.split('-', 2)
                if len(parts) >= 3:  # Format: WIDTHxHEIGHT-X-Y or WIDTHxHEIGHT+X-Y
                    x_part = parts[1]
                    if '+' in x_part:  # Handle case like WIDTHxHEIGHT+X-Y
                        x_parts = x_part.split('+')
                        if len(x_parts) >= 2:
                            x = int(x_parts[1])
                        else:
                            x = 100  # Default if parsing fails
                    else:
                        x = -int(x_part)  # Negative X
                    y = -int(parts[2])  # Negative Y
                else:
                    # Default values if parsing fails
                    x, y = 100, 100
            
            self.settings['window']['width'] = width
            self.settings['window']['height'] = height
            self.settings['window']['x'] = x
            self.settings['window']['y'] = y
            
            print(f"Saving window settings: {width}x{height} at position ({x},{y})")
            save_settings(self.settings)
        except Exception as e:
            print(f"Error saving window position: {e}")
            # Continue with shutdown even if saving settings fails
        
        # (Same as your previous version)
        print("Closing application requested...")
        self.monitoring_active = False
        self.stop_event.set()
        print("Waiting for data collector thread..."); self.data_collector_thread.join(timeout=UPDATE_INTERVAL_SECONDS + 1)
        print("Signaling CSV writer..."); self.csv_write_queue.put(None); self.csv_writer_thread.join(timeout=5)
        print("Waiting for periodic pruner thread..."); self.periodic_pruner_thread.join(timeout=5)
        self._perform_final_shutdown_tasks()
        
        # (Same as your previous version)
        print("Closing application requested...")
        self.monitoring_active = False
        self.stop_event.set()
        print("Waiting for data collector thread..."); self.data_collector_thread.join(timeout=UPDATE_INTERVAL_SECONDS + 1)
        print("Signaling CSV writer..."); self.csv_write_queue.put(None); self.csv_writer_thread.join(timeout=5)
        print("Waiting for periodic pruner thread..."); self.periodic_pruner_thread.join(timeout=5)
        self._perform_final_shutdown_tasks()

    def _perform_final_shutdown_tasks(self):
        # (Same as your previous version)
        print("Performing final shutdown tasks...")
        _prune_csv_file_core(CSV_FILENAME, DATA_WINDOW_HOURS, "Shutdown pruning")
        if NVML_INITIALIZED:
            try: pynvml.nvmlShutdown(); print("NVML Shutdown successful.")
            except pynvml.NVMLError as e: print(f"NVML Shutdown Error: {e}")
        
        # Check if the root window still exists before trying to destroy it
        try:
            if self.root.winfo_exists():
                print("Destroying Tkinter window..."); 
                self.root.destroy()
            else:
                print("Tkinter window already destroyed.")
        except Exception as e:
            print(f"Note: Could not destroy window: {e}")
            
        print("Application shutdown complete.")

# --- Main Execution ---
if __name__ == "__main__":
    initial_dataframe = load_initial_data(CSV_FILENAME, DATA_WINDOW_HOURS)
    root = tk.Tk()
    app = GpuMonitorApp(root, initial_dataframe)
    root.mainloop()
    print("Application mainloop finished.")