import imageio.v2 as imageio
from PIL import Image
from skimage.transform import rescale, resize

def load_image(path='data/phantom.png', resize_img = True):
    img =  imageio.imread(path, mode='L')
    if resize_img:
        return resize(img, (256, 256), mode='reflect', anti_aliasing=True)
    else:
        return img

def save_image(img, path='data/phantom_sinogram.png'):
    raw_img = Image.fromarray(img)
    raw_img = raw_img.convert("L")
    imageio.imwrite(path, raw_img)