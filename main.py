import pyautogui
import tkinter as tk
from PIL import PngImagePlugin, Image
from tkinter import messagebox

from model import run_test
import pandas as pd
import shutil
from csv_operations import csv_operations

RESULT_PATH = "./results_tst/stable_diffusion_256/stable_diffusion_256.csv"
CAPTURE_SAVE_PATH = "./TestSet/stable_diffusion_256"

def run_inference():
    print("Start inference")
    shutil.rmtree("./results_tst")

    # crop and resize image
    csv_operations("./TestSet", "./TestSetCSV", "./TestSet/operations.csv")

    # run inference
    run_test("./TestSetCSV/", "./results_tst", "./weights", "./TestSetCSV/operations.csv")
    print("End inference")

    # read result
    result = pd.read_csv("./results_tst/stable_diffusion_256/stable_diffusion_256.csv")
    diffusion_logit = result['Grag2021_latent'][0]
    gan_logit = result['Grag2021_progan'][0]
    is_fake_diffusion = diffusion_logit > 0
    is_fake_gan = gan_logit > 0
    is_fake = is_fake_diffusion or is_fake_gan
    result_msg = '이 사진은 ' + ('가짜' if is_fake else '진짜') + '입니다.' + '\n' + 'Diffusion: ' + str(diffusion_logit) + '\n' + 'GAN: ' + str(gan_logit) + '\n' + 'logit이 양수이면 가짜, 음수이면 진짜입니다.'
    messagebox.showinfo("결과", result_msg)
    return


class GlobalCoordinates:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("누가 만들었을까?")
        self.root.geometry("200x200")
        self.root.configure(bg="#333") 

        self.instruction_label = tk.Label(self.root, text="판별할 영역을 드래그 해주세요", font=("Helvetica", 16), background="#333", foreground="white")
        self.instruction_label.pack(pady=20)

        self.start_button = tk.Button(self.root, text="시작", command=self.start_drag, height=80, width=80)
        self.start_button.pack(pady=50, padx=50)

    def start_drag(self):
        self.root.withdraw()  # 시작 버튼을 누르면 기존 윈도우를 숨깁니다
        screen_width, screen_height = pyautogui.size()  # 화면 크기 가져오기

        self.drag_window = tk.Toplevel(self.root)
        self.drag_window.attributes("-alpha", 0.1)  # 투명도 설정
        self.drag_window.geometry(f"{screen_width}x{screen_height}+0+0")  # 화면 크기에 맞게 최대화

        self.canvas = tk.Canvas(self.drag_window, bg=None, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0

    def on_mouse_down(self, event):
        self.start_x, self.start_y = pyautogui.position()

    def on_mouse_drag(self, event):
        self.end_x, self.end_y = pyautogui.position()

    def on_mouse_up(self, event):
        self.end_x, self.end_y = pyautogui.position()

        self.drag_window.destroy()  # 드래그가 끝나면 투명한 윈도우를 닫습니다
        
        capture_image = pyautogui.screenshot( region=(int(self.start_x), int(self.start_y), int(abs(self.start_x-self.end_x)), int(abs(self.start_y-self.end_y))))
        self.root.deiconify()  # 기존 윈도우를 다시 표시합니다
        
        capture_image.save(f"{CAPTURE_SAVE_PATH}/ann000000000981.png")

        run_inference()

    def run(self):
        self.root.mainloop()

app = GlobalCoordinates()
app.run()

