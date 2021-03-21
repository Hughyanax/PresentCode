import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()

with writer.saving(fig, "writer_test.mp4", 100):  # 100指的dpi，dot per inch，表示清晰度

    plt.axis([0, 100, 0, 1])
    plt.ion()
    
    xs = [0, 0]
    ys = [1, 1]
    
    for i in range(100):
        y = np.random.random()
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = i
        ys[1] = y
        plt.plot(xs, ys)
        plt.pause(0.1)
        writer.grab_frame()