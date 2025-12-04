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
        self.all_poses = []  # Store all poses in drawing order

        self.canvas_height = 400
        self.canvas_width = 400

        self.label = tk.Label(master, text="User GUI", fg="#FFFFFF", font=("Arial", 32, "bold"))
        self.label.pack()

        self.canvas = tk.Canvas(master, bg="white", width=self.canvas_width, height=self.canvas_height, bd=0, highlightthickness=0, relief="flat")
        self.canvas.pack(pady=20, padx=20)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)
        
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, font=("Arial", 18))
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="Save as NPZ", command=self.save_poses, font=("Arial", 18))
        self.save_button.pack(side=tk.LEFT, padx=5)

    def _calculate_centered_coords(self, x, y):
        center_x = self.canvas_width / 2
        center_y = self.canvas_height / 2

        relative_x = x - center_x
        relative_y = center_y - y

        return relative_x, relative_y

    def start_draw(self, event):

        self.is_drawing = True
        self.current_segment = []

        self.last_x = event.x
        self.last_y = event.y
        self.canvas.create_line(self.last_x, self.last_y, self.last_x, self.last_y, fill="#000000", width=4, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="path")

        centered_x, centered_y = self._calculate_centered_coords(event.x, event.y)
        self.current_segment.append([round(centered_x), round(centered_y)])

    def draw(self, event):
        if self.is_drawing:
            x = max(0, min(event.x, self.canvas_width))
            y = max(0, min(event.y, self.canvas_height))

            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="#000000", width=4, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="path")
            self.last_x = x
            self.last_y = y  

            centered_x, centered_y = self._calculate_centered_coords(x, y)
            self.current_segment.append([round(centered_x), round(centered_y)])

    def stop_draw(self, event):
        if not self.is_drawing:
            return

        self.is_drawing = False

        # Convert all points in current segment to poses and add to all_poses
        for point in self.current_segment:
            x, z = point[0], point[1]
            pose = [x, z]
            self.all_poses.append(pose)
        
        print(f"Segment completed. Total poses: {len(self.all_poses)}") 

    def clear_canvas(self):
        self.canvas.delete("path")
        self.current_segment = []
        self.all_poses = []
        print("Canvas cleared")

    def save_poses(self):
        """Save all poses in drawing order as npz file"""
        if len(self.all_poses) == 0:
            print("No poses to save. Please draw something first.")
            return
        
        # Convert to numpy array: shape (N, 7)
        poses_array = np.array(self.all_poses, dtype=np.float32)
        
        # Default save directory: current directory
        default_dir = Path.cwd()
        default_filename = "canvas_poses.npz"
        default_path = default_dir / default_filename
        
        # Ask user for save location, with default path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            title="Save poses as NPZ",
            initialdir=str(default_dir),
            initialfile=default_filename
        )
        
        if file_path:
            # Ensure parent directory exists
            save_path = Path(file_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as npz file
            np.savez(save_path, poses=poses_array)
            abs_path = save_path.resolve()
            print(f"Saved {len(self.all_poses)} poses to: {abs_path}")
            print(f"Pose array shape: {poses_array.shape}")

    def get_paths(self):
        return self.current_segment
    
    def get_poses(self):
        """Get all poses in drawing order as numpy array"""
        if len(self.all_poses) == 0:
            return np.array([], dtype=np.float32).reshape(0, 7)
        return np.array(self.all_poses, dtype=np.float32)


if __name__ == '__main__':
    root = tk.Tk()
    app = Whiteboard(root)
    root.mainloop()