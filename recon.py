import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from scipy.fft import fft2, fftshift
from backprojection import backprojection, prep_image, chunk_recon
from image_io import load_image, save_image

if __name__=="__main__":
    img = load_image('data/phantom.png', True)
    sinogram = load_image('data/phantom_sinogram.png', False)
    theta = np.arange(0,180)
    chunk_width = 30
    
    #_, recon = backprojection(sinogram, theta=theta,filter = 'ramp', output_size=img.shape[0])
    
    filtered_sinogram = prep_image(sinogram, filter='ramp')
    output_size = 256

    recon = np.zeros((output_size, output_size),
                             dtype=filtered_sinogram.dtype)
    

    for x in range(6):
        start_theta = chunk_width*x
        end_theta = start_theta + chunk_width 
        theta = np.arange(start_theta, end_theta)

        chunk = filtered_sinogram[:, start_theta:end_theta]

        tmp_recon = chunk_recon(chunk, theta, output_size)
        recon = recon + tmp_recon

    # Display the original image, reconstruction, and Fourier Transform
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.set_title("Original Image")
    ax1.imshow(img, cmap=plt.cm.Greys_r)

    ax2.set_title("Sinogram")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r)

    ax3.set_title("Reconstruction\nFiltered back projection")
    ax3.imshow(np.abs(recon), cmap=plt.cm.Greys_r)

    plt.show()