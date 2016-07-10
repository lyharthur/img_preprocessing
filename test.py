from scipy import signal
from scipy import misc
import numpy as np
lena = misc.lena() - misc.lena().mean()
template = np.copy(lena[235:295, 310:370]) # right eye


corr = signal.correlate2d(lena, template, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
rowN ,colN = template.shape

print(y-rowN//2+1)
print(x-colN//2+1)

print(lena[y-rowN//2 + 1,x-colN//2 + 1]) #左上角的點
print(template)

