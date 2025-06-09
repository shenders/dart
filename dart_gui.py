import tkinter as tk
import subprocess
import json
import sys
if sys.version_info[:2] != (3, 7):
    print("⚠️  This script requires Python 3.7 due to outdated tkinter version 8.5. Please run it with Python 3.7.")
    sys.exit(1)
try:
    import ttkbootstrap as ttk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "ttkbootstrap"])
    import ttkbootstrap as ttk
from ttkbootstrap.style import Style
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from dart import dart
from scipy.interpolate import interp1d
import numpy as np
from load_plasma import plasma
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # Now we will remove the forward/backward buttons (the arrows)
        self._remove_buttons()

    def _remove_buttons(self):
        # Remove forward and backward navigation buttons
        for button in self.winfo_children():
            if isinstance(button, tk.Button):
                if "forward" in button.cget("text").lower() or "back" in button.cget("text").lower():
                    button.destroy()
    def set_message(self, s):
        """Override this method to prevent x, y coordinates from displaying."""
        pass  # Do nothing instead of displaying coordinates

class WaveformEditor:
    def __init__(self, parent,x,y):
        self.x = x
        self.y = y

        self.dragging_point = None

        # Set up figure
        fs = 8
        self.fig, self.ax = plt.subplots(figsize=(3, 3), dpi=100)
        self.line, = self.ax.plot(self.x, self.y, 'o-', picker=5)
        self.ax.set_xlabel("Time (s)",fontsize=fs)
        self.ax.set_ylabel("Confinement Time (ms)",fontsize=fs)
        self.ax.set_title("Left drag; Middle delete; Right add",fontsize=fs)
        self.ax.tick_params(axis='both',labelsize=fs)
        self.fig.tight_layout()  # ← This stops label overlap/cropping

        # Embed the plot in the provided parent frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Connect mouse interaction
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
        self.canvas.mpl_connect('button_press_event', self.on_middle_click)
    def on_pick(self, event):
        # Store index of picked point
        if event.artist != self.line:
            return
        mouse_event = event.mouseevent
        self.dragging_point = event.ind[0]
    def on_right_click(self, event):
        if event.button == 3 and event.inaxes:  # Right-click (button 3)
            if event.xdata is None or event.ydata is None:
                return  # Don't add point if clicked outside axes
            
            new_x = max(0, event.xdata)
            new_y = max(0, event.ydata)
            self.x = np.append(self.x, new_x)
            self.y = np.append(self.y, new_y)
            # Keep points sorted by x
            sorted_indices = np.argsort(self.x)
            self.x = self.x[sorted_indices]
            self.y = self.y[sorted_indices]

            self.update_plot()
    def on_middle_click(self, event):
        if event.button == 2 and event.inaxes:  # Middle-click (button 2)
            if event.xdata is None or event.ydata is None:
                return  # Don't delete point if clicked outside axes

            # Find the closest point to the click location
            distances = np.sqrt((self.x - event.xdata)**2 + (self.y - event.ydata)**2)
            closest_index = np.argmin(distances)

            # Set a threshold distance to ensure the click is close enough to a point
            threshold = 0.1  # Adjust this value as needed
            if distances[closest_index] < threshold:
                # Remove the point
                self.x = np.delete(self.x, closest_index)
                self.y = np.delete(self.y, closest_index)
                self.update_plot()
    def on_drag(self, event):
        if self.dragging_point is None or not event.inaxes:
            return

        # Update y-value only (confinement time)
        self.y[self.dragging_point] = max([0,event.ydata])
        self.x[self.dragging_point] = event.xdata
        # Sort by x to prevent crossing lines
        sorted_indices = np.argsort(self.x)
        self.x = self.x[sorted_indices]
        self.y = self.y[sorted_indices]
        self.dragging_point = sorted_indices.tolist().index(self.dragging_point)
        self.ax.set_xlim(min(self.x)*0.9, max(self.x)*1.1)
        self.ax.set_ylim(0, max(self.y)*1.1)
        self.update_plot()

    def on_release(self, event):
        self.dragging_point = None

    def update_plot(self):
        self.line.set_ydata(self.y)
        self.line.set_xdata(self.x)
        self.canvas.draw()

class DartGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DART Detachment Analysis with Reduced modelling Tools: Henderson et al. 2025 NF 65 016033 (DOI: 10.1088/1741-4326/ad93e7)")
        # Toolbar
        self.style = Style("litera")
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        # Create File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save session", command=self.save_session)
        file_menu.add_command(label="Load session", command=self.load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app)
        theme_menu = tk.Menu(menubar,tearoff=0)
        style = Style()
        available_themes = style.theme_names()
        for theme in available_themes:
            theme_menu.add_command(label=theme,command=lambda t=theme: self.change_theme(t))
                    
        # Add File menu to menubar
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Theme", menu=theme_menu)
        
        # Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Configure grid to make columns resizeable
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Setup notebook and tabs
        # Left side input tabs
        self.left_notebook = ttk.Notebook(self.left_frame)
        self.left_notebook.pack()
        self.aim_tab = ttk.Frame(self.left_notebook)
        self.aim_label = ttk.Label(self.aim_tab, text="Actuated Input Model (AIM)", font=("Arial", 10, "bold"))
        self.aim_label.grid(row=0,column=0, pady=3,sticky="W")
        self.left_notebook.add(self.aim_tab, text="AIM")
        #self.left_notebook.title("Actuation Input Model")
        self.guide_tab = ttk.Frame(self.left_notebook)
        self.guide_label = ttk.Label(self.guide_tab, text="Gas puffing Influence on Detachment Extent (GUIDE)", font=("Arial", 10, "bold"))
        self.guide_label.grid(row=0,column=0, pady=3,sticky="W")
        self.left_notebook.add(self.guide_tab, text="GUIDE")
        # Right side plot windows 
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack()

        self.standard_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.standard_tab, text="Plot panel 1")
        self.condensed_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.condensed_tab, text="Plot panel 2")
        self.useful_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.useful_tab, text="Plot panel 3")

        # GUIDE display        
        self.pulse_frame = tk.LabelFrame(self.guide_tab, text="Device details")
        self.pulse_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")
        tk.Label(self.pulse_frame, text="Device:").grid(row=0, column=0, sticky='w')
        self.device_options = ["MAST-U","AUG","JET"]
        self.device_selected_option = ttk.StringVar()
        self.dropdown_device = ttk.Combobox(self.pulse_frame,textvariable=self.device_selected_option,values=self.device_options,state="readonly")
        self.dropdown_device.grid(row=0, column=1, sticky="w")
        self.dropdown_device.current(0)
        tk.Label(self.pulse_frame, text="Shot:").grid(row=1, column=0, sticky='w')
        self.shot_entry = tk.Entry(self.pulse_frame, width=30)
        self.shot_entry.grid(row=1, column=1, sticky="w")
        # Pumping
        tk.Label(self.pulse_frame, text="Lower cryopump:").grid(row=2, column=0, sticky='w')
        self.cryo = tk.IntVar(value=1)
        self.cryo_checkbox = ttk.Checkbutton(self.pulse_frame, variable=self.cryo)
        self.cryo_checkbox.grid(row=2, column=1, sticky="nsew")

        # Gas valves
        self.valve_frame = tk.LabelFrame(self.guide_tab, text="Gas species")
        self.valve_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="nsew")
        # Common gas options
        gas_options = ['D', 'N']

        # Gas groups
        tk.Label(self.valve_frame, text="HFS-Mid").grid(row=1, column=0, sticky="w")
        self.hfs_gas_var = tk.StringVar(value=gas_options[0])
        self.hfs_gas_menu = ttk.Combobox(self.valve_frame, textvariable=self.hfs_gas_var, values=gas_options, state="readonly")
        self.hfs_gas_menu.grid(row=1, column=1, sticky="ew")

        tk.Label(self.valve_frame, text="LFS-V").grid(row=2, column=0, sticky="w")
        self.lfv_gas_var = tk.StringVar(value=gas_options[0])
        self.lfv_gas_menu = ttk.Combobox(self.valve_frame, textvariable=self.lfv_gas_var, values=gas_options, state="readonly")
        self.lfv_gas_menu.grid(row=2, column=1, sticky="ew")

        tk.Label(self.valve_frame, text="LFS-D").grid(row=3, column=0, sticky="w")
        self.lfd_gas_var = tk.StringVar(value=gas_options[0])
        self.lfd_gas_menu = ttk.Combobox(self.valve_frame, textvariable=self.lfd_gas_var, values=gas_options, state="readonly")
        self.lfd_gas_menu.grid(row=3, column=1, sticky="ew")

        tk.Label(self.valve_frame, text="LFS-S").grid(row=4, column=0, sticky="w")
        self.lfs_gas_var = tk.StringVar(value=gas_options[0])
        self.lfs_gas_menu = ttk.Combobox(self.valve_frame, textvariable=self.lfs_gas_var, values=gas_options, state="readonly")
        self.lfs_gas_menu.grid(row=4, column=1, sticky="ew")        

        tk.Label(self.valve_frame, text="PFR").grid(row=5, column=0, sticky="w")
        self.pfr_gas_var = tk.StringVar(value=gas_options[0])
        self.pfr_gas_menu = ttk.Combobox(self.valve_frame, textvariable=self.pfr_gas_var, values=gas_options, state="readonly")
        self.pfr_gas_menu.grid(row=5, column=1, sticky="ew")        

        # Plasma particle confinement input
        x=np.array([-100,15.0 ,20.0 ,110.0,300.0,400.0,1000.0,1100.0])
        y=np.array([0   ,0.0  ,2.0  ,2.0  ,3.0  ,3.0  ,0.8   ,0.0])
        self.wave_frame = tk.LabelFrame(self.guide_tab, text="Reservoir plasma particle confinement")
        self.wave_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky="nsew")
        self.editor = WaveformEditor(self.wave_frame,x/1000.0,y)
        self.wave_frame.grid_propagate(False)

        self.diffusion_frame = tk.LabelFrame(self.guide_tab, text="Reservoir diffusion inputs")
        self.diffusion_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky="nsew")
        tk.Label(self.diffusion_frame, text="Divertor-LFS time:").grid(row=0, column=0, sticky='w')
        self.closure_entry = tk.Entry(self.diffusion_frame, width=30)
        self.closure_entry.grid(row=0, column=1, sticky="w")
        self.closure_entry.insert(0, "0.4")
        tk.Label(self.diffusion_frame, text="Divertor-subdivertor time:").grid(row=1, column=0, sticky='w')
        self.div2sub_entry = tk.Entry(self.diffusion_frame, width=30)
        self.div2sub_entry.grid(row=1, column=1, sticky="w")
        self.div2sub_entry.insert(0, "0.5")

        self.recycling_frame = tk.LabelFrame(self.guide_tab, text="Plasma recycling fractions")
        self.recycling_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky="nsew")
        tk.Label(self.recycling_frame, text="Divertors:").grid(row=0, column=0, sticky='w')
        self.frac_div_entry = tk.Entry(self.recycling_frame, width=30)
        self.frac_div_entry.grid(row=0, column=1, sticky="w")
        self.frac_div_entry.insert(0, "0.1")
        tk.Label(self.recycling_frame, text="LFS:").grid(row=1, column=0, sticky='w')
        self.frac_lfs_entry = tk.Entry(self.recycling_frame, width=30)
        self.frac_lfs_entry.grid(row=1, column=1, sticky="w")
        self.frac_lfs_entry.insert(0, "0.12")
        
        # Run buttons
        self.shotrun_button = ttk.Button(self.guide_tab, text="Run", command=self.run_shot, style="success.TButton")
        style = ttk.Style()
        style.configure("success.TButton", font=("Arial", 12, "bold"), background="#28a745", foreground="white")
        self.shotrun_button.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")
        self.shotreplot_button = tk.Button(self.guide_tab, text="Replot", command=self.plot_shot)
        self.shotreplot_button.grid(row=7, column=0, columnspan=2, pady=5, sticky="nsew")
        self.shotreplot_button.grid_forget()  # Hide it initially
        self.shotprogress_bar = ttk.Progressbar(self.guide_tab, orient="horizontal", length=100, mode="determinate")
        self.shotprogress_bar.grid(row=8,column=0,columnspan=2, pady=5, sticky="nsew")

       
        # AIM display        
        # JETTO input
        self.jetto_var = tk.IntVar()
        self.jetto_check = ttk.Checkbutton(self.aim_tab, text="Load JETTO Alias", variable=self.jetto_var,command=self.toggle_widgets)
        self.jetto_check.grid(row=1, column=0, sticky="w")
        self.jetto_entry = tk.Entry(self.aim_tab, width=30)
        self.jetto_entry.grid(row=1, column=1, sticky="w")
        self.jetto_entry.insert(0, "feriks/jetto/step/88888/jun2623/seq-1")
        self.jetto_entry.config(state="disabled")
        self.jetto_var1 = tk.IntVar()
        self.jetto_check1 = ttk.Checkbutton(self.aim_tab, text="Append JETTO Alias", variable=self.jetto_var1,command=self.toggle_widgets_2)        
        self.jetto_check1.grid(row=2, column=0, sticky="w")
        self.jetto_entry1 = tk.Entry(self.aim_tab, width=30)
        self.jetto_entry1.grid(row=2, column=1, sticky="w")
        self.jetto_entry1.insert(0, "feriks/jetto/step/88888/oct3024/seq-1")
        self.jetto_entry1.config(state="disabled")
        self.jetto_check1.config(state="disabled")
            
        # Manual Input Frame
        self.manual_frame = tk.LabelFrame(self.aim_tab, text="Manual Input Waveforms")
        self.manual_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky="nsew")
        
        self.entries = {}
        self.labels = ["Time", "Plasma Current [MA]", "Fusion Power [GW]", "Auxiliary Heating [MW]", "Core frad", "Separatrix Density [1e19 m^-3]", "Target Grazing Angle [degrees]", "Input Detachment Qualifier","SOL width multiplier"]
        self.input_defaults = {
            "Time": "0.0,2.45,2.5,7.5,7.55,10.0",
            "Plasma Current [MA]": "2,20,21,21,20,2",
            "Fusion Power [GW]": "0.0,0.0,1.7,1.7,0.5,0.0",
            "Auxiliary Heating [MW]": "14,100,150,150,150,14",
            "Core frad": "0.3,0.3,0.7,0.7,0.5,0.3",
            "Separatrix Density [1e19 m^-3]": "1,3,4.2,4.2,4.2,1",
            "Target Grazing Angle [degrees]": "0.5,4.0,4.0,4.0,4.0,0.5",
            "Input Detachment Qualifier": "1.0,1.0,1.0,1.0,1.0,1.0",
            "SOL width multiplier": "1.0,1.0,1.0,1.0,1.0,1.0"
        }
        
        for i, label in enumerate(self.labels):
            tk.Label(self.manual_frame, text=label+":").grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.manual_frame)
            entry.insert(0, self.input_defaults[label])
            entry.grid(row=i, column=1, sticky="nsew")
            self.entries[label] = entry
        
        # Machine Details Frame
        self.machine_frame = tk.LabelFrame(self.aim_tab, text="Machine Details")
        self.machine_frame.grid(row=4, column=0, columnspan=6, pady=5, sticky="nsew")
        
        self.machine_entries = {}
        self.machine_labels = ["R0 [m]", "B0 [T]", "Amin [m]", "Elongation", "Target Radius [m]", "SOL Power Fraction", "Pump Speed [m^3/s]", "Wall Temperature [K]", "Div/sub DP","Impurity"]
        self.machine_defaults = ["3.6", "3.2", "2.0", "2.98", "5.6", "0.4", "20.0", "580.0","2.0","Ar"]
        
        for i, (label, default) in enumerate(zip(self.machine_labels, self.machine_defaults)):
            tk.Label(self.machine_frame, text=label+":").grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.machine_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky="nsew")
            self.machine_entries[label] = entry
        
        # Run Button
        self.run_button = ttk.Button(self.aim_tab, text="Run", command=self.run_simulation, style="success.TButton")
        style = ttk.Style()
        style.configure("success.TButton", font=("Arial", 12, "bold"), background="#28a745", foreground="white")
        self.run_button.grid(row=5, column=0, columnspan=2, pady=5, sticky="nsew")
        self.replot_button = tk.Button(self.aim_tab, text="Replot", command=self.plot_simulation)
        self.replot_button.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")
        self.replot_button.grid_forget()  # Hide it initially
        self.progress_bar = ttk.Progressbar(self.aim_tab, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.grid(row=7,column=0,columnspan=2, pady=5, sticky="nsew")
        self.step_button = tk.Button(self.aim_tab, text="Load STEP machine defaults", command=self.setup_step)
        self.step_button.grid(row=8, column=0, columnspan=2, pady=1, sticky="nsew")
        self.aug_button = tk.Button(self.aim_tab, text="Load AUG machine defaults", command=self.setup_aug)
        self.aug_button.grid(row=9, column=0, columnspan=2, pady=1, sticky="nsew")
        self.jet_button = tk.Button(self.aim_tab, text="Load JET machine defaults", command=self.setup_jet)
        self.jet_button.grid(row=10, column=0, columnspan=2, pady=1, sticky="nsew")
        self.mastu_button = tk.Button(self.aim_tab, text="Load MAST-U machine defaults", command=self.setup_mastu)
        self.mastu_button.grid(row=11, column=0, columnspan=2, pady=1, sticky="nsew")
        self.steptrace_button = tk.Button(self.aim_tab, text="Load STEP paper traces", command=self.setup_steptraces)
        self.steptrace_button.grid(row=12, column=0, columnspan=2, pady=1, sticky="nsew")
        
        # Plot Frame
        self.fig, self.ax = plt.subplots(figsize=(11.0,8.0))
        self.ax.text(0.5, 0.5, "Please click run to begin", 
                     fontsize=14, fontstyle='italic', ha='center', va='center', color='gray')

        self.ax.set_xticks([])  # Remove x-axis ticks
        self.ax.set_yticks([])  # Remove y-axis ticks
        self.ax.set_frame_on(False)  # Hide the plot border
        self.canvas_standard = FigureCanvasTkAgg(self.fig, master=self.standard_tab)
        toolbar = CustomToolbar(self.canvas_standard, self.standard_tab)
        toolbar.update()
        toolbar.grid(row=1, column=0,sticky="nsew")
        self.canvas_standard.get_tk_widget().grid()
        self.canvas_condensed = FigureCanvasTkAgg(self.fig, master=self.condensed_tab)
        toolbar = CustomToolbar(self.canvas_condensed, self.condensed_tab)
        toolbar.update()
        toolbar.grid(row=1, column=0,sticky="nsew")
        self.canvas_condensed.get_tk_widget().grid()
        self.canvas_useful = FigureCanvasTkAgg(self.fig, master=self.useful_tab)
        toolbar = CustomToolbar(self.canvas_useful, self.useful_tab)
        toolbar.update()
        toolbar.grid(row=1, column=0,sticky="nsew")
        self.canvas_useful.get_tk_widget().grid()
    def change_theme(self,theme_name):
        try:
            self.style.theme_use(theme_name)
            print(f"Theme changed to: {theme_name}")
        except Exception as e:
            messagebox.showerror("Theme error: ",f"Could not apply theme: {e}")
    def exit_app(self):
        root.quit()
    def save_session(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not filepath:
            return
        data = {
            "guide": {
                "shot": self.shot_entry.get(),
                "device": self.device_selected_option.get(),
                "gas_selection": {
                    "hfs": self.hfs_gas_var.get(),
                    "lfv": self.lfv_gas_var.get(),
                    "lfd": self.lfd_gas_var.get(),
                    "lfs": self.lfs_gas_var.get(),
                    "pfr": self.pfr_gas_var.get()
                },
                "conftime_x": self.editor.x.tolist(),
                "conftime_y": self.editor.y.tolist(),
                "closure": self.closure_entry.get(),
                "div2sub": self.div2sub_entry.get(),
                "frac_div": self.frac_div_entry.get(),
                "frac_lfs": self.frac_lfs_entry.get(),
                "cryo": self.cryo.get()
            },
            "aim":{
                "jetto_var": self.jetto_var.get(),
                "jetto_var1": self.jetto_var1.get(),
                "jetto_entry": self.jetto_entry.get(),
                "jetto_entry1": self.jetto_entry1.get(),
                "manual_inputs": {label: self.entries[label].get() for label in self.entries},
                "machine_inputs": {label: self.machine_entries[label].get() for label in self.machine_entries}
            }
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Save Session", f"Session inputs saved to:\n{filepath}")

    def load_session(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not filepath:
            return

        with open(filepath, "r") as f:
            data = json.load(f)
        aim = data.get("aim", {})
        # Load Jetto toggles
        self.jetto_var.set(aim.get("jetto_var", 0))
        self.jetto_var1.set(aim.get("jetto_var1", 0))
        self.toggle_widgets()
        self.toggle_widgets_2()

        self.jetto_entry.delete(0, tk.END)
        self.jetto_entry.insert(0, aim.get("jetto_entry", ""))

        self.jetto_entry1.delete(0, tk.END)
        self.jetto_entry1.insert(0, aim.get("jetto_entry1", ""))

        # Load manual waveform inputs
        manual_inputs = aim.get("manual_inputs", {})
        for label, val in manual_inputs.items():
            if label in self.entries:
                self.entries[label].delete(0, tk.END)
                self.entries[label].insert(0, val)

        # Load machine parameters
        machine_inputs = aim.get("machine_inputs", {})
        for label, val in machine_inputs.items():
            if label in self.machine_entries:
                self.machine_entries[label].delete(0, tk.END)
                self.machine_entries[label].insert(0, val)
        guide = data.get("guide", {})

        self.shot_entry.delete(0, tk.END)
        self.shot_entry.insert(0, guide.get("shot", ""))

        device = guide.get("device", self.device_options[0])
        if device in self.device_options:
            self.device_selected_option.set(device)

        self.hfs_gas_var.set(guide.get("gas_selection", {}).get("hfs", "D"))
        self.lfv_gas_var.set(guide.get("gas_selection", {}).get("lfv", "D"))
        self.lfd_gas_var.set(guide.get("gas_selection", {}).get("lfd", "D"))
        self.lfs_gas_var.set(guide.get("gas_selection", {}).get("lfs", "D"))
        self.pfr_gas_var.set(guide.get("gas_selection", {}).get("pfr", "D"))

        self.editor.x = np.array(guide.get("conftime_x", self.editor.x))
        self.editor.y = np.array(guide.get("conftime_y", self.editor.y))
        self.editor.update_plot()

        self.closure_entry.delete(0, tk.END)
        self.closure_entry.insert(0, guide.get("closure", "0.4"))

        self.div2sub_entry.delete(0, tk.END)
        self.div2sub_entry.insert(0, guide.get("div2sub", "0.5"))

        self.frac_div_entry.delete(0, tk.END)
        self.frac_div_entry.insert(0, guide.get("frac_div", "0.1"))

        self.frac_lfs_entry.delete(0, tk.END)
        self.frac_lfs_entry.insert(0, guide.get("frac_lfs", "0.12"))

        self.cryo.set(guide.get("cryo", 1))
        messagebox.showinfo("Load Session", f"Inputs loaded from:\n{filepath}")
        self.guide_tab.update_idletasks()
        self.aim_tab.update_idletasks()
    def toggle_widgets(self):
        machine_labels = ["R0 [m]", "B0 [T]", "Amin [m]", "Elongation"]        
        if self.jetto_var.get():  # If checkbox is selected
            for i, label in enumerate(self.labels):
               self.entries[label].config(state="disabled")
            for i, label in enumerate(machine_labels):
                self.machine_entries[label].config(state="disabled")
            self.jetto_check1.config(state="normal")
            self.jetto_entry.config(state="normal")
            if self.jetto_var1.get():  # If checkbox is selected
                self.jetto_entry1.config(state="normal")
        else:
            for i, label in enumerate(self.labels):
               self.entries[label].config(state="normal")
            for i, label in enumerate(machine_labels):
                self.machine_entries[label].config(state="normal")
            self.jetto_check1.config(state="disabled")
            self.jetto_entry.config(state="disabled")
            self.jetto_entry1.config(state="disabled")
    def toggle_widgets_2(self):
        if self.jetto_var1.get():  # If checkbox is selected
            self.jetto_entry1.config(state="normal")
        else:
            self.jetto_entry1.config(state="disabled")


    def update_machine_defaults(self):
        for i, (label, default) in enumerate(zip(self.machine_labels, self.machine_defaults)):
            tk.Label(self.machine_frame, text=label+":").grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.machine_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky="nsew")
            self.machine_entries[label] = entry
            self.root.update_idletasks()                
    def update_input_defaults(self):
        for i, label in enumerate(self.labels):
            tk.Label(self.manual_frame, text=label+":").grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.manual_frame)
            entry.insert(0, self.input_defaults[label])
            entry.grid(row=i, column=1, sticky="nsew")
            self.entries[label] = entry
            self.root.update_idletasks()                
        
    def setup_aug(self):
        self.machine_defaults = ["1.65", "2.5", "0.5", "1.6", "1.6", "0.5", "30.0", "580.0","1.0","Ar"]
        self.update_machine_defaults()
        self.input_defaults = {
            "Time": "0.0,1.0,3.0,5.0,7.0,9.0,10.0",
            "Plasma Current [MA]": "0.1,1.0,1.0,1.0,1.0,1.0,0.1",
            "Fusion Power [GW]": "0.0,0.0,0.0,0.0,0.0,0.0,0.0",
            "Auxiliary Heating [MW]": "0.1,5.0,12.0,12.0,12.0,5.0,0.1",
            "Core frad": "0.3,0.3,0.5,0.5,0.5,0.5,0.3",
            "Separatrix Density [1e19 m^-3]": "1,3.5,3.5,3.5,3.5,3.5,1",
            "Target Grazing Angle [degrees]": "2.0,2.5,2.5,2.5,2.5,2.5,2.0",
            "Input Detachment Qualifier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "SOL width multiplier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        }
        self.update_input_defaults()
    def setup_step(self):
        self.machine_defaults = ["3.6", "3.2", "2.0", "2.98", "5.6", "0.4", "24.0", "580.0","2.0","Ar"]
        self.update_machine_defaults()
        self.input_defaults = {
            "Time": "0.0,2.45,2.5,7.5,7.55,10.0",
            "Plasma Current [MA]": "2,20,21,21,20,2",
            "Fusion Power [GW]": "0.0,0.0,1.7,1.7,0.5,0.0",
            "Auxiliary Heating [MW]": "14,100,150,150,150,14",
            "Core frad": "0.3,0.3,0.7,0.7,0.5,0.3",
            "Separatrix Density [1e19 m^-3]": "1,3,4.2,4.2,4.2,1",
            "Target Grazing Angle [degrees]": "0.5,4.0,4.0,4.0,4.0,0.5",
            "Input Detachment Qualifier": "1.0,1.0,1.0,1.0,1.0,1.0",
            "SOL width multiplier": "1.0,1.0,1.0,1.0,1.0,1.0"
        }
        self.update_input_defaults()
    def setup_steptraces(self):
        self.machine_defaults = ["3.6", "3.2", "2.0", "2.98", "5.6", "0.4", "24.0", "580.0","2.0","Ar"]
        self.update_machine_defaults()
        self.input_defaults = {
            "Time": "0.0   , 2.45   , 2.5    , 3.5    , 3.7    , 4.6    , 4.8    , 5.0    , 6.0    , 6.2    , 6.4    , 7.3   , 7.5    , 7.55   , 10.0",
            "Plasma Current [MA]": "2.0 , 20.0 , 21.0 , 21.0 , 21.0 , 21.0 , 21.0 , 21   , 21   , 21   , 21   , 21  , 21   , 20.0 , 2.0",
            "Fusion Power [GW]": "0.0   , 0.0    , 1.7  , 1.7  , 1.4  , 1.4  , 1.7  , 1.7  , 1.7  , 1.7  , 1.7  , 1.7 , 1.7  , 0.5  , 0.0 ",
            "Auxiliary Heating [MW]": "14.0, 100.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150  , 150  , 150 , 150.0, 150.0, 14.0",
            "Core frad": "0.3   , 0.3    , 0.7    , 0.7    , 0.7    , 0.7    , 0.7    , 0.6    , 0.6    , 0.7    , 0.7    , 0.7   , 0.7    , 0.5    , 0.3  ",
            "Separatrix Density [1e19 m^-3]": "1.0, 3.0 , 4.2 , 4.2 , 5.8 , 5.8 , 4.2 , 4.7 , 4.7 , 4.2 , 4.2 , 4.2, 4.2 , 4.2 , 1.0 ",
            "Target Grazing Angle [degrees]": "0.5   , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0    , 4.0   , 4.0    , 4.0    , 0.5",
            "Input Detachment Qualifier": "1.0   , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 5.0    , 5.0   , 1.0    , 1.0    , 1.0  ",
            "SOL width multiplier": "1.0   , 1.0    , 1.0    , 1.0    , 0.3    , 0.3   , 1.0    , 1.0    , 1.0    , 1.0    , 1.0    , 1.0   , 1.0    , 1.0    , 1.0  "
        }
        self.update_input_defaults()
        
    def setup_jet(self):
        self.machine_defaults = ["2.9", "2.5", "0.9", "1.6", "2.9", "0.5", "130.0", "580.0","1.5","Ar"]
        self.update_machine_defaults()
        self.input_defaults = {
            "Time": "0.0,1.0,3.0,5.0,7.0,9.0,10.0",
            "Plasma Current [MA]": "0.1,1.0,2.5,2.5,2.5,1.0,0.1",
            "Fusion Power [GW]": "0.0,0.0,0.0,0.0,0.0,0.0,0.0",
            "Auxiliary Heating [MW]": "0.1,5.0,25.0,25.0,25.0,5.0,0.1",
            "Core frad": "0.3,0.3,0.5,0.5,0.5,0.3,0.3",
            "Separatrix Density [1e19 m^-3]": "0.5,1.5,3.0,3.0,3.0,1.5,0.5",
            "Target Grazing Angle [degrees]": "2.0,2.0,3.0,3.0,3.0,2.0,2.0",
            "Input Detachment Qualifier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "SOL width multiplier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        }
        self.update_input_defaults()
    def setup_mastu(self):
        self.machine_defaults = ["0.85", "0.55", "0.5", "2.1", "0.7", "0.3", "50.0", "580.0","1.0","N"]
        self.update_machine_defaults()
        self.input_defaults = {
            "Time": "0.0,0.1,0.3,0.5,0.7,0.9,1.0",
            "Plasma Current [MA]": "0.1,0.75,0.75,0.75,0.75,0.75,0.1",
            "Fusion Power [GW]": "0.0,0.0,0.0,0.0,0.0,0.0,0.0",
            "Auxiliary Heating [MW]": "0.1,1.8,3.5,3.5,3.5,3.5,0.1",
            "Core frad": "0.2,0.2,0.2,0.2,0.2,0.2,0.2",
            "Separatrix Density [1e19 m^-3]": "0.3,1.0,1.0,1.0,1.0,1.0,0.3",
            "Target Grazing Angle [degrees]": "2.0,8.0,8.0,8.0,8.0,8.0,2.0",
            "Input Detachment Qualifier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "SOL width multiplier": "1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        }
        self.update_input_defaults()
    def run_shot(self):
        plasma_details = plasma(shot=int(self.shot_entry.get()),
                                device=self.device_selected_option.get(),                                
                                progress_bar=self.shotprogress_bar,
                                root=self.root)
        self.dart = dart()
        self.dart.time = plasma_details.time
        self.dart.Ip   = plasma_details.Ip
        self.dart.pvol = plasma_details.vol
        self.dart.Paux = plasma_details.Paux
        self.dart.Pfus = plasma_details.Paux * 0.0
        self.dart.frad = plasma_details.frad
        self.dart.IRt2 = plasma_details.IRt2
        self.dart.IRt5 = plasma_details.IRt5
        self.dart.Ploss= plasma_details.Paux
        self.dart.Psep = plasma_details.Psep
        self.dart.drsep= plasma_details.drsep
        self.dart.fuelgas = plasma_details.gas
        self.dart.impgas  = plasma_details.imp
        self.dart.gas_matrix = {
                'HFS': 1 if self.hfs_gas_var.get() == 'N' else 0,
                'LFS': 1 if self.lfv_gas_var.get() == 'N' else 0,
                'UDV': 1 if self.lfd_gas_var.get() == 'N' else 0,
                'UDVS': 1 if self.lfs_gas_var.get() == 'N' else 0,
                'LDV': 1 if self.lfd_gas_var.get() == 'N' else 0,
                'LDVS': 1 if self.lfs_gas_var.get() == 'N' else 0,
                'LPFR': 1 if self.pfr_gas_var.get() == 'N' else 0,
                'UPFR': 1 if self.pfr_gas_var.get() == 'N' else 0
             }

        self.dart.shot = int(self.shot_entry.get())
        self.dart.dens = plasma_details.dens
        self.dart.Rt   = plasma_details.Rt
        self.dart.R0   = plasma_details.R0
        self.dart.am   = plasma_details.am
        self.dart.B0   = plasma_details.B0
        self.dart.kp   = plasma_details.kp
        self.dart.press= plasma_details.p0
        self.dart.p0mid= plasma_details.p0mid
        self.dart.p0up = plasma_details.p0up
        self.dart.closure=float(self.closure_entry.get())
        self.dart.div2sub=float(self.div2sub_entry.get())
        hfs_frac       = 1.0 - 0.01 - float(self.frac_div_entry.get()) - float(self.frac_lfs_entry.get())
        self.dart.plasma_fracs=[float(self.frac_div_entry.get()),float(self.frac_lfs_entry.get()),hfs_frac,0.01,0.0]
        self.dart.alft = np.radians(plasma_details.alft)  
        conftime       = self.editor.x 
        conf           = self.editor.y 
        pconftime      = interp1d(conftime,conf/1000.0,bounds_error=False, fill_value=0.0)
        self.dart.plasma_conf = pconftime(self.dart.time)
        if self.cryo.get():
            self.dart.Spump = 10.0
        else:
            self.dart.Spump = 0.0                
        self.dart.Twall = 300.0
        self.dart.r_fit     = plasma_details.r_fit
        self.dart.te_fit    = plasma_details.te_fit
        self.dart.ne_fit    = plasma_details.ne_fit
        self.dart.runGUIDE()
        self.shotprogress_bar["value"] = 90.0
        self.root.update_idletasks()  # Refresh GUI

        self.plot_shot()
        self.shotreplot_button.grid(row=7, column=0, columnspan=2, pady=5, sticky="nsew")
        self.shotprogress_bar["value"] = 100.0
        self.root.update_idletasks()  # Refresh GUI
    def run_simulation(self):
        if self.jetto_var.get():
            alias0 = self.jetto_entry.get()
            if not alias0:
                messagebox.showerror("Error", "Please enter a Jetto alias.")
                return
            if self.jetto_var1.get():
                alias1 = self.jetto_entry1.get()
                if not alias1:
                    messagebox.showerror("Error", "Please enter a 2nd Jetto alias.")
                    return
                alias = [alias0,alias1]
            else:
                alias = alias0
            self.dart = dart(jetto=alias)
        
        else:
            tarr = np.array(list(map(float, self.entries["Time"].get().split(','))))
            Ip_vals = np.array(list(map(float, self.entries["Plasma Current [MA]"].get().split(','))))
            Pfus_vals = np.array(list(map(float, self.entries["Fusion Power [GW]"].get().split(','))))
            Paux_vals = np.array(list(map(float, self.entries["Auxiliary Heating [MW]"].get().split(','))))
            frad_vals = np.array(list(map(float, self.entries["Core frad"].get().split(','))))
            nsep_vals = np.array(list(map(float, self.entries["Separatrix Density [1e19 m^-3]"].get().split(','))))
            alft_vals = np.radians(np.array(list(map(float, self.entries["Target Grazing Angle [degrees]"].get().split(',')))))
            qdet_vals = np.array(list(map(float, self.entries["Input Detachment Qualifier"].get().split(','))))
            sol_vals  = np.array(list(map(float, self.entries["SOL width multiplier"].get().split(','))))
            
            self.dart = dart()
            self.dart.time = self.dart.log_vector(np.min(tarr), np.max(tarr), num=50, linear=True)
            self.dart.Ip, self.dart.Pfus, self.dart.Paux, self.dart.frad, self.dart.nsep, self.dart.alft, self.dart.qdet0,self.dart.solm = [
                np.interp(self.dart.time, tarr, vals) for vals in [Ip_vals*1e6, Pfus_vals*1e9, Paux_vals*1e6, frad_vals, nsep_vals*1e19, alft_vals, qdet_vals, sol_vals]]
            self.dart.tilt = np.radians(1.5)
            self.dart.Ploss = self.dart.Pfus/5.0 + self.dart.Paux
            self.dart.Psep  = self.dart.Ploss * (1-self.dart.frad)
            self.dart.R0, self.dart.B0, self.dart.am, self.dart.kp = float(self.machine_entries["R0 [m]"].get()), float(self.machine_entries["B0 [T]"].get()), float(self.machine_entries["Amin [m]"].get()), float(self.machine_entries["Elongation"].get())

        self.dart.imp = str(self.machine_entries["Impurity"].get())
        self.dart.fdiv = float(self.machine_entries["SOL Power Fraction"].get())
        self.dart.Rt, self.dart.Spump, self.dart.Twall, self.dart.dp = float(self.machine_entries["Target Radius [m]"].get()), float(self.machine_entries["Pump Speed [m^3/s]"].get()), float(self.machine_entries["Wall Temperature [K]"].get()), float(self.machine_entries["Div/sub DP"].get())            
        self.dart.run(progress_bar=self.progress_bar,root=self.root)
        self.plot_simulation()
        self.replot_button.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")
    def plot_simulation(self):
        self.dart.display(canvas=self.canvas_standard)
        self.dart.display_condensed(canvas=self.canvas_condensed)
        self.dart.display_useful(canvas=self.canvas_useful)
    def plot_shot(self):
        self.dart.plot_GUIDE(canvas=self.canvas_standard)
        self.dart.plot_GUIDEcondensed(canvas=self.canvas_condensed)
        self.dart.plot_GUIDEuseful(canvas=self.canvas_useful)

if __name__ == "__main__":
    root = ttk.Window(themename="litera")
    app = DartGUI(root)
    root.geometry("1600x900")  # Adjust window size as needed
    root.mainloop()
