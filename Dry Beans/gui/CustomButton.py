import tkinter as tk


class CustomButton:
    def __init__(self, frame, text):
        self.button_border = tk.Frame(frame, bg="white")
        self.bar_btn = tk.Button(
            frame,
            relief=tk.FLAT,
            text=text,
            bg="#262730",
            font=("bold", 16),
            fg="white",
            cursor="hand2",
            activebackground="#262730",
            activeforeground="#dc4343",
        )
        self.bar_btn.bind('<Enter>', lambda event: self.on_enter(self.bar_btn))
        self.bar_btn.bind('<Leave>', lambda event: self.on_leave(self.bar_btn))
        self.bar_btn.bind("<Configure>", self.adjust_border_size)

    def place(self, x, y):
        self.button_border.place(x=x-1, y=y-1)
        self.bar_btn.place(x=x, y=y)

    def on_enter(self, e):
        e.config(foreground="#dc4343")
        self.button_border.config(background="#dc4343")

    def on_leave(self, e):
        e.config(foreground="white")
        self.button_border.config(background="white")

    def adjust_border_size(self, event):
        self.button_border.configure(width=self.bar_btn.winfo_width() + 2, height=self.bar_btn.winfo_height() + 2)
