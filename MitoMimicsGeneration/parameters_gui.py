import argparse
import copy
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import yaml

from parameter_descriptions import get_description


DEFAULT_SIM_PATH = Path("parameters.yaml")
DEFAULT_RENDER_PATH = Path("parameters_render.yaml")
DEFAULT_SIM_DEFAULTS_PATH = Path("default_parameters.yaml")
DEFAULT_RENDER_DEFAULTS_PATH = Path("default_parameters_render.yaml")
PADDING_X = 8
PADDING_Y = 4
WRAP_LENGTH = 420


class YamlParameterEditor:
    def __init__(
        self,
        root: tk.Tk,
        sim_path: Path,
        render_path: Path,
        sim_defaults_path: Path,
        render_defaults_path: Path,
    ) -> None:
        self.root = root
        self.root.title("MitoMimics Parameter Editor")
        self.root.geometry("1400x850")

        self.file_paths = {
            "sim": Path(sim_path),
            "render": Path(render_path),
        }
        self.default_file_paths = {
            "sim": Path(sim_defaults_path),
            "render": Path(render_defaults_path),
        }

        self.reference_defaults = {
            name: self._load_yaml(path)
            for name, path in self.default_file_paths.items()
        }
        self.current_data = {
            name: self._load_yaml(path)
            for name, path in self.file_paths.items()
        }
        self.original_loaded_data = copy.deepcopy(self.current_data)

        self.selected_script = tk.StringVar(value="sim")
        self.status_var = tk.StringVar(value="Ready")
        self.description_var = tk.StringVar(value="Select a parameter to see its description.")
        self.input_vars: dict[str, tk.Variable] = {}

        self._build_layout()
        self._render_editor("sim")

    def _build_layout(self) -> None:
        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="Editing:").pack(side="left")
        ttk.Radiobutton(
            top_bar,
            text="Simulation parameters",
            variable=self.selected_script,
            value="sim",
            command=self._on_switch_script,
        ).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(
            top_bar,
            text="Render parameters",
            variable=self.selected_script,
            value="render",
            command=self._on_switch_script,
        ).pack(side="left", padx=4)

        ttk.Button(top_bar, text="Reload from disk", command=self._reload_from_disk).pack(side="right", padx=4)
        ttk.Button(top_bar, text="Save current file", command=self._save_current_file).pack(side="right", padx=4)

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        editor_container = ttk.Frame(main)
        desc_container = ttk.Frame(main, padding=10)
        main.add(editor_container, weight=5)
        main.add(desc_container, weight=2)

        self.canvas = tk.Canvas(editor_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(editor_container, orient="vertical", command=self.canvas.yview)
        self.form_frame = ttk.Frame(self.canvas)
        self.form_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._resize_canvas_window)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        ttk.Label(desc_container, text="Parameter description", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.desc_name_label = ttk.Label(desc_container, text="No parameter selected", font=("TkDefaultFont", 10, "bold"))
        self.desc_name_label.pack(anchor="w", pady=(10, 6))
        self.desc_value_label = ttk.Label(
            desc_container,
            textvariable=self.description_var,
            wraplength=WRAP_LENGTH,
            justify="left",
        )
        self.desc_value_label.pack(anchor="nw", fill="x")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken", padding=(8, 4))
        status_bar.pack(fill="x", side="bottom")

    def _resize_canvas_window(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _load_yaml(self, path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _save_yaml(self, path: Path, data) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)

    def _on_switch_script(self) -> None:
        self._render_editor(self.selected_script.get())

    def _reload_from_disk(self) -> None:
        script_name = self.selected_script.get()
        try:
            self.current_data[script_name] = self._load_yaml(self.file_paths[script_name])
            self.original_loaded_data[script_name] = copy.deepcopy(self.current_data[script_name])
            self.reference_defaults[script_name] = self._load_yaml(self.default_file_paths[script_name])
            self._render_editor(script_name)
            self.status_var.set(
                f"Reloaded current file {self.file_paths[script_name].name} and defaults {self.default_file_paths[script_name].name}"
            )
        except Exception as exc:
            messagebox.showerror("Reload failed", str(exc))

    def _save_current_file(self) -> None:
        script_name = self.selected_script.get()
        try:
            updated_data = self._collect_values(script_name)
            self.current_data[script_name] = updated_data
            self.original_loaded_data[script_name] = copy.deepcopy(updated_data)
            self._save_yaml(self.file_paths[script_name], updated_data)
            self.status_var.set(f"Saved {self.file_paths[script_name].name}")
            messagebox.showinfo("Saved", f"Saved {self.file_paths[script_name].name}")
            self._render_editor(script_name)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _render_editor(self, script_name: str) -> None:
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.input_vars.clear()

        current_data = self.current_data[script_name]
        defaults_data = self.reference_defaults[script_name]

        ttk.Label(
            self.form_frame,
            text=f"{script_name.upper()} parameters",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        ttk.Label(
            self.form_frame,
            text=(
                f"Editing file: {self.file_paths[script_name].name}\n"
                f"Defaults file: {self.default_file_paths[script_name].name}"
            ),
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        self._build_group(
            parent=self.form_frame,
            script_name=script_name,
            current_group=current_data,
            defaults_group=defaults_data,
            parent_path="",
        )
        self.canvas.yview_moveto(0)
        self.desc_name_label.config(text="No parameter selected")
        self.description_var.set("Select a parameter to see its description.")

    def _build_group(self, parent, script_name: str, current_group: dict, defaults_group: dict, parent_path: str) -> None:
        keys = list(current_group.keys())
        for key in defaults_group.keys():
            if key not in current_group:
                keys.append(key)

        for key in keys:
            path = f"{parent_path}.{key}" if parent_path else key
            current_value = current_group.get(key)
            default_value = defaults_group.get(key)

            if isinstance(current_value, dict) or isinstance(default_value, dict):
                nested_current = current_value if isinstance(current_value, dict) else {}
                nested_default = default_value if isinstance(default_value, dict) else {}
                group = ttk.LabelFrame(parent, text=key, padding=10)
                group.pack(fill="x", expand=True, padx=10, pady=8, anchor="n")
                self._build_group(group, script_name, nested_current, nested_default, path)
            else:
                self._build_parameter_row(parent, script_name, key, path, current_value, default_value)

    def _build_parameter_row(self, parent, script_name: str, key: str, path: str, current_value, default_value) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="x", expand=True, padx=10, pady=5)
        outer.columnconfigure(1, weight=1)

        label = ttk.Label(outer, text=key, width=35)
        label.grid(row=0, column=0, sticky="nw", padx=(0, 8))

        editor_frame = ttk.Frame(outer)
        editor_frame.grid(row=0, column=1, sticky="ew")
        editor_frame.columnconfigure(0, weight=1)

        widget, variable = self._make_input_widget(editor_frame, current_value, default_value)
        widget.grid(row=0, column=0, sticky="ew")
        self.input_vars[path] = variable

        button_frame = ttk.Frame(outer)
        button_frame.grid(row=0, column=2, sticky="ne", padx=(8, 0))

        ttk.Button(
            button_frame,
            text="Reset to default",
            command=lambda p=path, s=script_name: self._reset_parameter_to_default(s, p),
        ).pack(anchor="e")

        ttk.Button(
            button_frame,
            text="?",
            width=3,
            command=lambda p=path, s=script_name: self._show_description(s, p),
        ).pack(anchor="e", pady=(4, 0))

        current_text = self._value_to_text(self._lookup_value(self.original_loaded_data[script_name], path, missing_text="<missing>"))
        default_text = self._value_to_text(default_value)

        meta = ttk.Frame(outer)
        meta.grid(row=1, column=1, sticky="ew", pady=(3, 0))

        ttk.Label(meta, text=f"Current file value: {current_text}").pack(anchor="w")
        ttk.Label(meta, text=f"Default value: {default_text}").pack(anchor="w")

        for target in (label, widget):
            target.bind("<FocusIn>", lambda e, p=path, s=script_name: self._show_description(s, p))
            target.bind("<Button-1>", lambda e, p=path, s=script_name: self._show_description(s, p))

    def _make_input_widget(self, parent, current_value, default_value):
        value_for_widget = current_value if current_value is not None else default_value

        if isinstance(value_for_widget, bool):
            var = tk.BooleanVar(value=bool(value_for_widget))
            widget = ttk.Checkbutton(parent, variable=var)
            return widget, var

        var = tk.StringVar(value=self._value_to_text(value_for_widget))
        widget = ttk.Entry(parent, textvariable=var)
        return widget, var

    def _value_to_text(self, value) -> str:
        if value is None:
            return "null"
        if isinstance(value, (list, dict)):
            return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
        return str(value)

    def _parse_input_value(self, raw_value: str):
        try:
            parsed = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse value '{raw_value}': {exc}") from exc

        if parsed is None and raw_value.strip().lower() not in {"null", "none", "~", ""}:
            return raw_value
        return parsed

    def _lookup_value(self, data, path: str, missing_text=None):
        current = data
        for part in path.split('.'):
            if not isinstance(current, dict) or part not in current:
                return missing_text
            current = current[part]
        return current

    def _assign_value(self, data: dict, path: str, value) -> None:
        parts = path.split('.')
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _collect_values(self, script_name: str):
        updated = copy.deepcopy(self.current_data[script_name])
        for path, variable in self.input_vars.items():
            raw_default = self._lookup_value(self.reference_defaults[script_name], path)
            current_raw = variable.get()
            if isinstance(variable, tk.BooleanVar):
                parsed_value = bool(current_raw)
            else:
                parsed_value = self._parse_input_value(str(current_raw))
            self._assign_value(updated, path, parsed_value)

            if raw_default is None and self._lookup_value(self.reference_defaults[script_name], path, missing_text="__missing__") == "__missing__":
                # keep keys editable even if only present in current file
                continue
        return updated

    def _reset_parameter_to_default(self, script_name: str, path: str) -> None:
        default_value = self._lookup_value(self.reference_defaults[script_name], path, missing_text="__missing__")
        if default_value == "__missing__":
            messagebox.showerror(
                "Default missing",
                f"No default value was found for '{path}' in {self.default_file_paths[script_name].name}.",
            )
            return

        variable = self.input_vars[path]
        if isinstance(variable, tk.BooleanVar):
            variable.set(bool(default_value))
        else:
            variable.set(self._value_to_text(default_value))

        self.status_var.set(f"Reset {path} to default from {self.default_file_paths[script_name].name}")
        self._show_description(script_name, path)

    def _show_description(self, script_name: str, path: str) -> None:
        self.desc_name_label.config(text=f"{script_name}: {path}")
        current_value = self.input_vars.get(path)
        current_value_text = ""
        if current_value is not None:
            current_value_text = self._value_to_text(current_value.get())
        default_value = self._lookup_value(self.reference_defaults[script_name], path, missing_text="<missing>")
        loaded_value = self._lookup_value(self.original_loaded_data[script_name], path, missing_text="<missing>")
        description = get_description(script_name, path)
        self.description_var.set(
            f"{description}\n\n"
            f"Current editor value: {current_value_text}\n"
            f"Current file value: {self._value_to_text(loaded_value)}\n"
            f"Default value: {self._value_to_text(default_value)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple Tkinter GUI for editing YAML parameters.")
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM_PATH, help="Path to the editable simulation YAML file.")
    parser.add_argument("--render", type=Path, default=DEFAULT_RENDER_PATH, help="Path to the editable render YAML file.")
    parser.add_argument(
        "--sim-defaults",
        type=Path,
        default=DEFAULT_SIM_DEFAULTS_PATH,
        help="Path to the immutable default simulation YAML file.",
    )
    parser.add_argument(
        "--render-defaults",
        type=Path,
        default=DEFAULT_RENDER_DEFAULTS_PATH,
        help="Path to the immutable default render YAML file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    missing_paths = [
        path for path in [args.sim, args.render, args.sim_defaults, args.render_defaults]
        if not path.exists()
    ]
    if missing_paths:
        missing_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"These YAML files were not found:\n{missing_str}")

    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    app = YamlParameterEditor(
        root=root,
        sim_path=args.sim,
        render_path=args.render,
        sim_defaults_path=args.sim_defaults,
        render_defaults_path=args.render_defaults,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
