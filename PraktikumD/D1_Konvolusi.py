import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_noise(image, noise_type):
    if noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], :] = 1

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
        noisy = image + gauss
        return noisy
    elif noise_type == "spike":
        # Spike noise could be implemented similarly to salt and pepper but with larger spikes
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

    return output


# Contoh penggunaan
# Membaca citra asli
image = cv2.imread('image_with_noise.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Menambahkan noise pada citra
noisy_image = add_noise(image, "salt_pepper")

# Menampilkan citra asli dan citra dengan noise
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(noisy_image), plt.title('Image with Salt & Pepper Noise')
plt.show()

# Definisi kernel (misal kernel blur)
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0

# Melakukan konvolusi
convolved_image = convolve2D(noisy_image, kernel)

# Menampilkan citra hasil konvolusi
plt.imshow(convolved_image.astype(np.uint8))
plt.title('Convolved Image')
plt.show()