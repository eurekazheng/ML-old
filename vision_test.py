import visdom
from PIL import Image

img = Image.open('sample/10_left.jpeg')
viz = visdom.Visdom()
viz.image(img)