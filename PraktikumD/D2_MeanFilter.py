import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, Label, Button, filedialog, OptionMenu, StringVar
from PIL import Image, ImageTk


def add_noise(image, noise_type):
    if noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], :] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1], :] = 0
        return out
    elif noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 255
        noisy = np.clip(noisy, 0, 255)  # Ensure values are within [0, 255]
        return noisy.astype(np.uint8)
    elif noise_type == "spike":
        row, col, ch = image.shape
        amount = 0.004
        out = np.copy(image)
        num_spike = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_spike)) for i in image.shape]
        out[coords[0], coords[1], :] = 255  # Assuming white spikes
        return out
    else:
        return image


def convolve2D(image, kernel):
    i_height, i_width = image.shape[:2]
    k_height, k_width = kernel.shape[:2]
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros on the border
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant',
                          constant_values=0)

    output = np.zeros((i_height, i_width, 3))

    for i in range(pad_height, i_height + pad_height):
        for j in range(pad_width, i_width + pad_width):
            for k in range(3):  # Assuming the image has 3 channels (RGB)
                region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1, k]
                output[i - pad_height, j - pad_width, k] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)  # Ensure values are within [0, 255]


def apply_filter(image, filter_type):
    if filter_type == "mean":
        kernel = np.ones((3, 3), np.float32) / 9
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:
        return image
    return convolve2D(image, kernel)


def load_image():
    global original_image
    file_path = filedialog.askopenfilename()
    original_image = cv2.imread(file_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    display_image(original_image, "Original Image")


def display_image(image, title):
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel = Label(root, image=imgtk)
    panel.image = imgtk
    panel.grid(row=1, column=0, columnspan=3)
    title_label = Label(root, text=title)
    title_label.grid(row=0, column=0, columnspan=3)


def process_images():
    global original_image
    noise_type = noise_var.get()
    filter_type = filter_var.get()

    noisy_images = []
    for i in range(5):
        noisy_image = add_noise(original_image, noise_type)
        noisy_images.append(noisy_image)
        plt.subplot(2, 5, i + 1)
        plt.imshow(noisy_image)
        plt.title(f'Noisy Image {i + 1} ({noise_type})')

    # Menampilkan citra asli
    plt.subplot(2, 5, 6)
    plt.imshow(original_image)
    plt.title('Original Image')

    # Melakukan smoothing atau sharpening pada citra dengan noise dan menampilkan hasilnya
    for i in range(5):
        processed_image = apply_filter(noisy_images[i], filter_type)
        plt.subplot(2, 5, i + 6)
        plt.imshow(processed_image)
        plt.title(f'Processed Image {i + 1}')

    plt.show()


root = Tk()
root.title("Image Processing GUI")

original_image = None

load_button = Button(root, text="Load Image", command=load_image)
load_button.grid(row=2, column=0)

noise_var = StringVar(root)
noise_var.set("salt_pepper")
noise_menu = OptionMenu(root, noise_var, "salt_pepper", "gaussian", "spike")
noise_menu.grid(row=2, column=1)

filter_var = StringVar(root)
filter_var.set("mean")
filter_menu = OptionMenu(root, filter_var, "mean", "sharpen")
filter_menu.grid(row=2, column=2)

process_button = Button(root, text="Process Images", command=process_images)
process_button.grid(row=3, column=0, columnspan=3)

root.mainloop()
