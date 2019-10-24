from skimage.transform import resize, rotate
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import sys

def discrete_radon_transform(image, steps):
    image = resize(image, (steps,steps))
    R = np.zeros((steps, len(image)), dtype='float64')
    print(R.shape)
    for s in range(steps):
        rotation = rotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R


def get_fourier_filter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                        np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(np.fft.fft(f))         # ramp filter

    return fourier_filter[:, np.newaxis]

def inverse_radon(radon_image, ):

    theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)
    
    angles_count = len(theta)
    img_shape = radon_image.shape[0]
    
    output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = get_fourier_filter(projection_size_padded)
    projection = np.fft.fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(np.fft.ifft(projection, axis=0)[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    
    for col, angle in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        reconstructed += np.interp(t,xp=x, fp=col, left=0, right=0)

    return reconstructed * np.pi / (2 * angles_count)

def main(argv):
    image = imread(argv[1], as_gray = True)
    image_radon = discrete_radon_transform(image, 200)
    plt.imshow(image_radon)
    plt.savefig("output_radon.png")

    image_restored = inverse_radon(image_radon)
    plt.imshow(image_restored)
    plt.savefig("output_restored.png")
    return 0

if __name__ == "__main__":
    main(sys.argv)



