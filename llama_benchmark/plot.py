import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100, 400, 50)

# 设置画布的宽度
plt.figure(figsize=(12, 6))

# subplot 1
plt.subplot(1, 2, 1)
y_wait_on = [488.45630860328674, 736.6374139785767, 973.7714612483978,
             1190.081399679184, 1435.234134197235, 1639.3765404224396]

y_wait_off = [302.58964193662007, 408.6374139785767, 536.4381279150646,
              676.2147330125174, 812.0341341972352, 906.8432070891064]

plt.plot(x, y_wait_on, label='include interval')
plt.plot(x, y_wait_off, label='inference only')

plt.scatter(x, y_wait_on)
plt.scatter(x, y_wait_off)

# # Add labels at each point
# for (i, j) in zip(x, y_wait_on):
#         plt.text(i, j, f'({i},{j:0.2f})')
# for (i, j) in zip(x, y_wait_off):
#         plt.text(i, j, f'({i},{j:0.2f})')

plt.xlabel('max_request')
plt.ylabel('latency (ms)')
plt.title('preferred batch size = 20')
plt.legend()

# subplot 2
plt.subplot(1, 2, 2)
y_batchsize_20 = [302.58964193662007, 408.6374139785767, 536.4381279150646,
              676.2147330125174, 812.0341341972352, 906.8432070891064]

y_batchsize_50 = [236.6579903920492, 289.32206010818487, 376.13472167650866,
                  487.1783534844717, 580.5599273204805, 635.6084909598034]

# 设置y轴的限制为0-1640
plt.ylim(0, 1640)

plt.plot(x, y_batchsize_20, label='preferred_batch_size=20', color='sandybrown')
plt.plot(x, y_batchsize_50, label='preferred_batch_size=50', color='red')

plt.scatter(x, y_batchsize_20, color='sandybrown')
plt.scatter(x, y_batchsize_50, color='red')

plt.xlabel('max_request')
plt.ylabel('latency (ms)')
plt.title('Comparison')
plt.legend()

plt.savefig(f'./plot_result/latency_temp.png')