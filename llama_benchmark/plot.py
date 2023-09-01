import matplotlib.pyplot as plt
import numpy as np

preferred_batch_size = [1, 2, 4, 8, 16, 20, 25, 30, 32, 35, 40]
queue = [405708002.718, 25714416.976, 29124352.31, 34914620.006, 21843064.843,
         5818080.978, 7906248.725, 8321263.035, 8817501.383, 10649387.658,
         51684601.697]
infer = [7101166.739, 19520480.601, 17685265.024, 13460317.43, 16982682.393,
         24010376.812, 21073908.527, 21098667.411, 21115331.908, 19330868.564,
         21464537.25]
infer_per_req = [7101166.739, 19520480.601/2, 17685265.024/4, 13460317.43/8, 16982682.393/16,
                 5818080.978/20, 7906248.725/25, 8321263.035/30, 8817501.383/32, 10649387.658/35,
                 51684601.697/40]

x = np.array(preferred_batch_size[1:])
y = np.array(queue[1:])/1000000
y2 = np.array(infer[1:])/1000000
y3 = np.array(infer_per_req[1:])/1000000

# 设置画布大小
plt.figure(figsize=(12, 8))

# 绘制折线图
plt.plot(x, y, marker='o', color='r', label='Queue')
plt.plot(x, y2, marker='o', color='b', label='Inference')
plt.plot(x, y3, marker='o', color='g', label='Inference per request')

# 设置坐标轴的标签
plt.xlabel('preferred_batch_size')
plt.ylabel('latency (s)')

plt.legend()  # 显示图例

plt.savefig(f'./plot_result/latency_temp.png')