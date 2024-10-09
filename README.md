# ComfyUI_LG_FFT
Implementation of Fast Fourier Transform in COMFYUI

这是一个复刻PS FFT插件功能的节点

主要代码参考了https://github.com/fssorc/ComfyUI_FFT

在原代码的基础上增加了高通、低通、带通滤波器以及遮罩反转，提供了更多的灵活性，能够对不同程度的图片做进一步的调整以达到最佳效果

low_pass 低通滤波，保留低于cutoff的频率，仅需调整high_cutoff参数

high_pass 高通滤波，保留高于cutoff的频率，仅需调整low_cutoff参数

band_pass 带通滤波，保留low_cutoff和high_cutoff区间的频率，high_cutoff和low_cutoff 同时作用

常用修复网纹使用低通滤波，其他两个滤波方式也保留下来，也许会有特别的用途

# 该节点可与原作者fssorc的FFT节点通用，在此感谢原作者的无私分享！
![workflow](https://github.com/user-attachments/assets/03760628-3fde-46d7-bf23-5a0fcc746939)
![87206374f4e2c6a6439311cd81ca5ff](https://github.com/user-attachments/assets/09d9d6f2-a62a-47db-8e00-686582544d1b)
