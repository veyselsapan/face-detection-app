"""
This is the entry point of the application. 
It will handle the high-level flow of the program, such as initializing the application and orchestrating the use of other modules.
"""
import ui
import tkinter as tk
def main():
    ui.Application(tk.Tk(), "face detect")

if __name__ == "__main__":
    main()