# ========================================
# ||||||||||||| Imports  |||||||||||||||||
# ========================================

import subprocess
import sys
import tkinter as tk
from tkinter import colorchooser, ttk

import yaml
from PIL import ImageColor

from parameter_labels import (color_parameters, group_label_map, group_map, label_map, tooltips)

# ===========================================
# ||||||||||||| Global Functions  |||||||||||
# ===========================================

# Save new parameter values to "parameter.yaml" when Save button is clicked
def save_parameters():
    updated_parameters = {}
    shape_dict = {}
    for key, entry in entry_widgets.items():
        if isinstance(entry, tk.BooleanVar) and key[-1].isdigit():
            # Check for parameters with multiple boolean checkboxes (e.g. shape_choice)
            
            # Retrieve prefix of dictionary values i.e. shape_choices_blob_1 is shape_choices_blob
            key_prefix = key.rpartition('_')[0]
            heading = key_prefix.rpartition('_')[0]
            shape = key_prefix.rpartition('_')[2]

            dict_items = {dic_key: val for dic_key, val in entry_widgets.items() if dic_key.startswith(heading)}
            dict_items[shape] = dict_items.pop(key)
            shape_dict[shape] = dict_items[shape].get()

            updated_parameters[heading] = shape_dict
        elif isinstance(entry, tk.BooleanVar):
            # For boolean parameters, get the value directly from the BooleanVar
            updated_parameters[key] = entry.get()
        elif key[-1].isdigit(): 
            # Check if the parameter is a list item (color or range)

            # Retrieve the prefix of list entries as these are ascending numerical suffix entries
            key_prefix = key.rpartition('_')[0]
            # Prefix key match in dictionary
            list_items = {dic_key: val for dic_key, val in entry_widgets.items() if dic_key.startswith(key_prefix)}

            values_list = list()
            for list_box_num, list_box_val in list_items.items():
                # Check if list values is integer or float
                value = list_box_val.get()
                if value.isdigit():
                    # Check if the value can be converted to an integer
                    try:
                        value = int(value)
                    except ValueError:
                        # If conversion fails, keep the value as a string
                        pass
                    values_list.append(value)
                else:
                    value = float(value)
                    values_list.append(value)
            updated_parameters[key_prefix] = values_list

        else:
            value = entry.get()
            if value.isdigit():
                # Check if the value can be converted to an integer
                try:
                    value = int(value)
                except ValueError:
                    # If conversion fails, keep the value as a string
                    pass
                updated_parameters[key] = value
            elif '.' in value or 'e' in value:
                # Check if the value can be converted to a float
                try:
                    value = float(value)
                except ValueError:
                    # If conversion fails, keep the value as a string
                    pass
                updated_parameters[key] = value
            else:
                # If none of the above conditions are met, keep the value as a string
                updated_parameters[key] = value
    
    # Update the parameters dictionary with the new values
    parameters.update(updated_parameters)

    # Create a new dictionary to hold the grouped parameters
    grouped_parameters = {}
    for group, params in group_map.items():
        group_params = {}
        for param in params:
            if param in updated_parameters:
                group_params[param] = updated_parameters[param]
        if group_params:
            grouped_parameters[group] = group_params

    # Write the grouped parameters to the YAML file
    with open('parameters.yaml', 'w') as file:
        yaml.dump(grouped_parameters, file)

    message_label.config(text="Parameters saved successfully!")
    message_label.config(foreground="green")

# Create color picker widgets
def choose_color(label_text, entry_widgets):
    color = colorchooser.askcolor(title=f"Choose {label_text} color")[1]
    if color:
        # Convert hexadecimal color to RGB tuple
        rgb_tuple = ImageColor.getcolor(color, "RGB")
        rgb_list = list(rgb_tuple)
        # Update the entry widgets with the RGB values
        for i, value in enumerate(rgb_list):
            entry_widget = entry_widgets[i]
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, str(value))

# Hover over paramter names to display tooltip
def show_tooltip(widget, text):
    def on_enter(event):
        tooltip_label.config(text=text, foreground='white')
    def on_leave(event):
        tooltip_label.config(text="")
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

# Function to update the scroll region
def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

# Define scroll speed
def on_canvas_scroll(event):
    scroll_smooth = 1
    if event.delta > 0:
        canvas.yview_scroll(-scroll_smooth, "units")
    elif event.delta < 0:
        canvas.yview_scroll(scroll_smooth, "units")

# Execute "python sim.py" when Run Simulation button is clicked
def run_simulation():
    python_executable = sys.executable
    subprocess.call([python_executable, './sim.py'])

def toggle_group(group):
    if group["collapsed"]:
        group["frame"].pack(fill="y")
        group["collapsed"] = False
    else:
        group["frame"].pack_forget()
        group["collapsed"] = True

def toggle_group(self):
        if bool(self.show.get()):
            self.sub_frame.pack(fill="x", expand=1)
            self.toggle_button.configure(text='-')
        else:
            self.sub_frame.forget()
            self.toggle_button.configure(text='+')

# ============================================
# |||||||||||||||| GUI Main ||||||||||||||||||
# ============================================

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


## -------------- Initialise GUI ------------------
root = tk.Tk()
root.title('Parameter Editor')
# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Set the initial size of the window
root.geometry(f"{screen_width}x{screen_height}+0+0")


## -------------- Create GUI Frames ------------------

# Create a frame for the groups
main_frame = ttk.Frame(root) # maybe rename to main_frame
main_frame.pack(fill='both', expand=True)

# Create a frame for the groups
groups_frame = ttk.Frame(main_frame)
groups_frame.pack(side='left', fill='y')

# Create a frame for the Save button
save_button_frame = ttk.Frame(groups_frame)
save_button_frame.pack(side='bottom', fill='x', pady=10)

# Create a frame for the second frame to the right
right_frame = ttk.Frame(main_frame, borderwidth=2, relief='ridge')
right_frame.pack(side='right', fill='both', expand=True)

# Create a canvas for the scrollable area
canvas = tk.Canvas(groups_frame)
canvas.pack(side='left', fill='y')

# Add a scrollbar to the group_frame canvas
scrollbar = ttk.Scrollbar(groups_frame, orient='vertical', command=canvas.yview)
scrollbar.pack(side='right', fill='y')
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame to contain the group frames
group_container = ttk.Frame(canvas)
canvas.create_window((0, 0), window=group_container, anchor='nw')

# Bind the update_scroll_region function to the canvas
group_container.bind("<Configure>", update_scroll_region)


## -------------- Assign Parameters and Groups to Frames and Subdivide ------------------

# Create group, labels and entry widgets for each parameter grouped under different subheadings
group_frames = {}
collapsible_groups = {}
entry_widgets = {}
color_entries = []
for i, (group, params) in enumerate(group_map.items()):
    # Create frames for each group of parameters
    group_label = group_label_map.get(group, group.replace('_', ' ').title())
    group_frame = ttk.LabelFrame(group_container, text=group_label)
    group_frame.pack(fill='both', expand=True, padx=5, pady=5)
    group_frames[group] = group_frame

    for key in params:
        label_text = label_map.get(key, key.replace('_', ' ').title())
        label = ttk.Label(group_frame, text=label_text)
        label.pack(anchor='w', padx=5, pady=5)

        # Allow tooltips to generate for each parameter
        tooltip_text = tooltips.get(key, '')
        show_tooltip(label, tooltip_text)
        
        if isinstance(parameters[group][key], bool):
            # If the parameter is boolean, create a checkbox
            var = tk.BooleanVar(value=parameters[group][key])  # Use BooleanVar with initial value
            checkbox = ttk.Checkbutton(group_frame, variable=var)
            checkbox.pack(anchor='w', padx=5, pady=5)
            entry_widgets[key] = var  # Store BooleanVar object
        elif isinstance(parameters[group][key], dict):
            # If the parameter is a dictionary of booleans values assigned to a shape, create a checkbox
            i = 1
            for shape, values in parameters[group][key].items():
                var = tk.BooleanVar(value=values)
                checkbox = ttk.Checkbutton(group_frame, text=shape, variable=var)
                checkbox.pack(anchor='w', padx=5, pady=5)
                entry_widgets[f"{key}_{shape}_{i}"] = var  # Store BooleanVar object
                i += 1 # Tracker for saving
        elif isinstance(parameters[group][key], list) and key not in color_parameters:
            # For parameters that are lists and not a color parameter, create multiple entry widgets for each value
            range_entry_frame = ttk.Frame(group_frame)
            range_entry_frame.pack(anchor='w', padx=5, pady=5)

            for i, value in enumerate(parameters[group][key]):
                entry = ttk.Entry(range_entry_frame, width=3)
                entry.insert(tk.END, str(value))
                entry.pack(side='left', padx=5, pady=5)
                entry_widgets[f"{key}_{i+1}"] = entry
        elif key in color_parameters:
            # For color parameters that lists three RGB values, create three entry widgets for each value
            color_entry_frame = ttk.Frame(group_frame)
            color_entry_frame.pack(anchor='w', padx=5, pady=5)

            color_entry_widgets = []
            default_color = parameters[group].get(key, [255, 255, 255])  # Default to white if not found
            for i in range(3):
                color_entry = ttk.Entry(color_entry_frame, width=3)
                color_entry.insert(tk.END, str(default_color[i]))  # Insert default RGB values
                color_entry.pack(side='left', padx=5, pady=5)
                color_entry_widgets.append(color_entry)
                entry_widgets[f"{key}_{i+1}"] = color_entry
            color_entries.append(color_entry_widgets)
            
            # Update the lambda function for the color picker button
            color_picker_button = ttk.Button(group_frame, text='Choose Color',
                                  command=lambda rgb_entries=color_entry_widgets: choose_color(label_text, rgb_entries))
            
            color_picker_button.pack(anchor='w', padx=5, pady=5)
        
        else:
            # For other types (non-boolean and non-list), create a single entry widget
            entry = ttk.Entry(group_frame, width=6)
            if key in parameters[group]:
                entry.insert(tk.END, str(parameters[group][key]))
            entry.pack(anchor='w', padx=5, pady=5)
            entry_widgets[key] = entry


# Add the canvas to the scrollable area
canvas.update_idletasks()  # Update the canvas to calculate the scroll region
canvas.config(scrollregion=canvas.bbox("all"))

# Bind the update_scroll_region function to the canvas
for group_frame in group_frames.values():
    group_frame.bind("<MouseWheel>", on_canvas_scroll)


# ==========================================================
# |||||||||||||||| GUI Buttons & Messages ||||||||||||||||||
# ==========================================================

# Create a label for displaying tooltips
tooltip_label = ttk.Label(root, text="", foreground="black")
tooltip_label.pack()

# Create a Save button
save_button = ttk.Button(save_button_frame, text='Save', command=save_parameters)
save_button.pack(pady=10)

# Create a Simulation run button
run_simulation_button = ttk.Button(save_button_frame, text='Run Simulation', command=run_simulation)
run_simulation_button.pack(pady=10)

# Create a label for displaying messages
message_label = ttk.Label(root, text="", foreground="black")
message_label.pack()

# Start the GUI event loop
root.mainloop()