import tkinter as tk



class Whiteboard:

    def __init__(self, master):
        self.master = master

        master.title("DRACO")
        self.is_drawing = False
        self.current_segment = []

        self.canvas_height = 400
        self.canvas_width = 800

        self.label = tk.Label(master, text="User GUI", fg="#000000", font=("Arial", 32, "bold"))
        self.label.pack()

        self.canvas = tk.Canvas(master, bg="white", width=self.canvas_width, height=self.canvas_height, bd=4, relief="sunken")
        self.canvas.pack(pady=20, padx=20)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas, font=("Arial", 18))
        self.clear_button.pack(pady=10)

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
            x, y = event.x, event.y

            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="#000000", width=4, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="path")
            self.last_x = x
            self.last_y = y  

            centered_x, centered_y = self._calculate_centered_coords(x, y)
            self.current_segment.append([round(centered_x), round(centered_y)])

    def stop_draw(self, event):
        if not self.is_drawing:
            return

        self.is_drawing = False

        print(self.current_segment) 

    def clear_canvas(self):
        self.canvas.delete("path")
        self.current_segment = []
        print("Canvas cleared")


    def get_paths(self):
        return self.current_segment

if __name__ == '__main__':
    root = tk.Tk()
    app = Whiteboard(root)
    root.mainloop()