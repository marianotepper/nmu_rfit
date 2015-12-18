import tempfile
import os.path
import subprocess
import numpy as np


class Segment(object):
    def __init__(self, p_a, p_b, quality=None, width=None, precision=None):
        self.p_a = p_a
        self.p_b = p_b
        self.quality = quality
        self.width = width
        self.precision = precision
        line_ab = np.cross(p_a, p_b)
        self.line = line_ab / np.linalg.norm(line_ab)

    def plot(self, **kwargs):
        plt.plot([self.p_a[0], self.p_b[0]], [self.p_a[1], self.p_b[1]],
                 **kwargs)


def compute(gray_image, epsilon=1):
    fobj = tempfile.NamedTemporaryFile(suffix='.pgm')
    gray_image.save(fobj.name)
    
    exe = './lsd'
    if not os.path.exists(exe):
        sp = subprocess.Popen(['make'], stdout=open(os.devnull, 'wb'))
        sp.wait()

    fobj_txt = tempfile.NamedTemporaryFile(suffix='.txt')
    sp = subprocess.Popen([exe, fobj.name, fobj_txt.name])
    sp.wait()
    
    segments = []
    for line in fobj_txt:
        l = line.split(' ')
        values = [float(s) for s in l if s and s != '\n']
        segments.append(Segment(values[0:2], values[2:4], width=values[4],
                                precision=values[5], quality=values[6]))

    return segments


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import PIL.Image
    dir_name = '/Users/mariano/Documents/datasets/YorkUrbanDB/'
    img_name = dir_name + 'P1020839/P1020839.jpg'
    gray_image = PIL.Image.open(img_name).convert('L')
    segments = compute(gray_image, 1)

    plt.figure()
    plt.axis('off')
    plt.imshow(gray_image, cmap='gray', alpha=.5)
    for seg in segments:
        seg.plot(c='g', linewidth=1)

    plt.show()
