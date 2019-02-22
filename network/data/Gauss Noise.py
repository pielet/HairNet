import os
from skimage import io, util
import PythonMagick as PM

rootdir = os.path.join(os.getcwd(), 'ppm_images')
output_dir = os.path.join(os.getcwd(), 'x_png')

ppm_list = os.listdir(rootdir)
for infile in ppm_list:
    inpath = os.path.join(rootdir, infile)
    fname, suff = os.path.splitext(infile)
    outpath = os.path.join(output_dir, fname + '.png')
    PM.Image(inpath).write(outpath)
    img = io.imread(outpath)
    noisy_img = util.random_noise(img, mode="gaussian")
    io.imsave(outpath, noisy_img)
