import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DrawingClassifierApp:

    def __init__(self):
        self.class1_name, self.class2_name, self.class3_name = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.classifier = None
        self.project_name = None
        self.root = None
        self.image = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15

        self.prompt_project_details()
        self.init_gui()

    def prompt_project_details(self):
        msg = Tk()
        msg.withdraw()

        self.project_name = simpledialog.askstring("Project Name", "Please enter your project name:", parent=msg)
        if os.path.exists(self.project_name):
            self.load_data()
        else:
            self.prompt_class_names(msg)

    def prompt_class_names(self, msg):
        self.class1_name = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
        self.class2_name = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
        self.class3_name = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

        self.class1_counter = 1
        self.class2_counter = 1
        self.class3_counter = 1

        self.classifier = LinearSVC()

        os.mkdir(self.project_name)
        os.chdir(self.project_name)
        os.mkdir(self.class1_name)
        os.mkdir(self.class2_name)
        os.mkdir(self.class3_name)
        os.chdir("..")

    def load_data(self):
        with open(f"{self.project_name}/{self.project_name}_data.pickle", "rb") as f:
            data = pickle.load(f)
        self.class1_name = data['class1_name']
        self.class2_name = data['class2_name']
        self.class3_name = data['class3_name']
        self.class1_counter = data['class1_counter']
        self.class2_counter = data['class2_counter']
        self.class3_counter = data['class3_counter']
        self.classifier = data['classifier']
        self.project_name = data['project_name']

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"Drawing Classifier - Project: {self.project_name}")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image)

        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        class1_btn = Button(btn_frame, text=self.class1_name, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2_name, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3_name, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        bm_btn = Button(btn_frame, text="Brush-", command=self.brush_minus)
        bm_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear_canvas)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brush_plus)
        bp_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        change_btn = Button(btn_frame, text="Change Model", command=self.change_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.classifier).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, class_num):
        img = self._save_image()
        if class_num == 1:
            img.save(f"{self.project_name}/{self.class1_name}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.project_name}/{self.class2_name}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.project_name}/{self.class3_name}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1

        self.clear_canvas()

    def brush_minus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brush_plus(self):
        self.brush_width += 1

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.project_name}/{self.class1_name}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.project_name}/{self.class2_name}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.project_name}/{self.class3_name}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)

        self.classifier.fit(img_list, class_list)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully trained!")

    def predict(self):
        img = self._save_image()
        img.save("predict_shape.png", "PNG")

        img = cv.imread("predict_shape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.classifier.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class1_name}")
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class2_name}")
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class3_name}")

    def _save_image(self):
        self.image.save("temp.png")
        result = PIL.Image.open("temp.png")
        result.thumbnail((50, 50), PIL.Image.ANTIALIAS)
        return result

    def change_model(self):
        if isinstance(self.classifier, LinearSVC):
            self.classifier = KNeighborsClassifier()
        elif isinstance(self.classifier, KNeighborsClassifier):
            self.classifier = LogisticRegression()
        elif isinstance(self.classifier, LogisticRegression):
            self.classifier = DecisionTreeClassifier()
        elif isinstance(self.classifier, DecisionTreeClassifier):
            self.classifier = RandomForestClassifier()
        elif isinstance(self.classifier, RandomForestClassifier):
            self.classifier = GaussianNB()
        elif isinstance(self.classifier, GaussianNB):
            self.classifier = LinearSVC()

        self.status_label.config(text=f"Current Model: {type(self.classifier).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.classifier, f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully saved!")

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.classifier = pickle.load(f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully loaded!")

    def save_everything(self):
        data = {"class1_name": self.class1_name, "class2_name": self.class2_name, "class3_name": self.class3_name,
                "class1_counter": self.class1_counter, "class2_counter": self.class2_counter,
                "class3_counter": self.class3_counter, "classifier": self.classifier, "project_name": self.project_name}
        with open(f"{self.project_name}/{self.project_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Drawing Classifier", "Project successfully saved!")

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save your work?")
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

DrawingClassifierApp()
