import tkinter as tk
from tkinter import filedialog
import numpy as np
from pathlib import Path

class Whiteboard:
    def __init__(self, master):
        self.master = master

        master.title("DRACO")
        self.is_drawing = False
        self.current_segment = []
        self.all_poses = []  # Store all poses in drawing order (list of strokes)
        self.index = 1

        self.canvas_height = 400
        self.canvas_width = 400

        # Title label
        self.label = tk.Label(master, text="User GUI", fg="#FFFFFF", font=("Arial", 32, "bold"))
        self.label.pack()

        # Main drawing canvas
        self.canvas = tk.Canvas(
            master,
            bg="white",
            width=self.canvas_width,
            height=self.canvas_height,
            bd=0,
            highlightthickness=0,
            relief="flat"
        )
        self.canvas.pack(pady=20, padx=20)

        # Mouse bindings for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)
        
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, font=("Arial", 18))
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="Save as NPZ", command=self.save_poses, font=("Arial", 18))
        self.save_button.pack(side=tk.LEFT, padx=5)

    def _calculate_centered_coords(self, x, y):
        """Convert raw canvas coords to centered coords with origin at canvas center."""
        center_x = self.canvas_width / 2
        center_y = self.canvas_height / 2

        # x: right positive, y: up positive
        relative_x = x - center_x
        relative_y = center_y - y

        return relative_x, relative_y

    def start_draw(self, event):
        """Start a new stroke when mouse is pressed."""
        self.is_drawing = True
        self.current_segment = []

        self.last_x = event.x
        self.last_y = event.y

        # Draw a tiny starting point so stroke appears immediately
        self.canvas.create_line(
            self.last_x,
            self.last_y,
            self.last_x,
            self.last_y,
            fill="#000000",
            width=4,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            tags="path"
        )

        # Store first point in centered coords
        centered_x, centered_y = self._calculate_centered_coords(event.x, event.y)
        self.current_segment.append([round(centered_x), round(centered_y)])

    def draw(self, event):
        """Track mouse movement and draw line segments while mouse is pressed."""
        if self.is_drawing:
            # Clamp to canvas bounds
            x = max(0, min(event.x, self.canvas_width))
            y = max(0, min(event.y, self.canvas_height))

            # Draw segment from last point to current point
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                fill="#000000",
                width=4,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                tags="path"
            )

            self.last_x = x
            self.last_y = y  

            # Save pose in centered coordinates
            centered_x, centered_y = self._calculate_centered_coords(x, y)
            self.current_segment.append([round(centered_x), round(centered_y)])

    def stop_draw(self, event):
        """Finish the current stroke when mouse is released."""
        if not self.is_drawing:
            return

        self.is_drawing = False

        if not self.current_segment:
            print("Segment completed but no points recorded.")
            return

        # Append this stroke to global list
        self.all_poses.append(list(self.current_segment))
        print(f"Segment completed. Total segments: {len(self.all_poses)}")

    def clear_canvas(self):
        """Clear canvas and reset all stored strokes."""
        self.canvas.delete("path")
        self.current_segment = []
        self.all_poses = []
        self.index = 1
        print("Canvas cleared")

    def save_poses(self):
        """Save all recorded strokes as individual NPZ files under traces/<trace_name>/."""
        if len(self.all_poses) == 0:
            print("No poses to save. Please draw something first.")
            return

        trace_name = input("Enter trace name: ").strip()
        if not trace_name:
            print("Save canceled. Empty name.")
            return

        # traces/<trace_name>/<segment_index>_cposes.npz
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent / "traces" 
        save_dir = base_dir / trace_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, segment in enumerate(self.all_poses, start=1):
            poses_array = np.array(segment, dtype=np.float32)
            filename = f"{i:03d}_cposes.npz"
            save_path = save_dir / filename

            np.savez(save_path, poses=poses_array)
            print(f"Saved stroke {i} with {poses_array.shape[0]} poses to: {save_path}")

        print(f"Finished saving {len(self.all_poses)} strokes to {save_dir}")
        
        # Save last used trace name so the pipeline can pick it up
        trace_file = base_dir / ".last_trace"
        with open(trace_file, 'w') as f:
            f.write(trace_name)
        print(f"Trace name '{trace_name}' saved to {trace_file}")
        
        # Reset GUI after saving
        self.clear_canvas()

    def get_paths(self):
        """Return the current active stroke (not all strokes)."""
        return self.current_segment
    
    def get_poses(self):
        """Return all saved strokes as a list of float32 numpy arrays."""
        if len(self.all_poses) == 0:
            return []
        return [np.array(segment, dtype=np.float32) for segment in self.all_poses]


if __name__ == '__main__':
    root = tk.Tk()
    app = Whiteboard(root)
    root.mainloop()