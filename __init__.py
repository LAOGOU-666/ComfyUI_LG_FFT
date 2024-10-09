import torch
import torchvision.transforms.v2 as T
import numpy as np
import cv2

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)

class LG_FFTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FFTData")
    RETURN_NAMES = ("image", "FFTData")
    FUNCTION = "toFFT"
    CATEGORY = "🎈LAOGOU"

    def toFFT(self, image):
        FFTImageList = []
        FFT_Channel_Data = []
        channelCount = 3
        imageCount = image.shape[0]

        for i in range(image.shape[0]):
            sourceImg = image[i]
            cv2Image = (sourceImg.contiguous() * 255).byte()

            dim = sourceImg.dim()
            if dim == 3:
                R_channel = cv2Image[:, :, 0]
                G_channel = cv2Image[:, :, 1]
                B_channel = cv2Image[:, :, 2]

                fshiftData = []
                # 傅里叶变换
                R_fft = np.fft.fft2(R_channel)
                R_fshift = np.fft.fftshift(R_fft)
                fshiftData.append(R_fshift)
                G_fft = np.fft.fft2(G_channel)
                G_fshift = np.fft.fftshift(G_fft)
                fshiftData.append(G_fshift)
                B_fft = np.fft.fft2(B_channel)
                B_fshift = np.fft.fftshift(B_fft)
                fshiftData.append(B_fshift)

                R_img = np.log(np.abs(R_fshift))
                G_img = np.log(np.abs(G_fshift))
                B_img = np.log(np.abs(B_fshift))

                R_img = R_img / np.max(R_img)
                G_img = G_img / np.max(G_img)
                B_img = B_img / np.max(B_img)

                fftImg = np.dstack((R_img, G_img, B_img)).astype(np.float32)
                FFT_Channel_Data.append(fshiftData)
                FFTImageList.append(fftImg)
            else:
                channelCount = 1
                fshiftData = []
                # 单通道图像傅里叶变换
                R_fft = np.fft.fft2(cv2Image)
                R_fshift = np.fft.fftshift(R_fft)
                fshiftData.append(R_fshift)
                fftImg = np.log(np.abs(R_fshift))
                fftImg = fftImg / np.max(fftImg)
                fftImg = fftImg.astype(np.float32)
                FFTImageList.append(fftImg)
                FFT_Channel_Data.append(fshiftData)

        tensors_out = torch.stack([torch.from_numpy(np_array) for np_array in FFTImageList])
        FFT_Data = {'channelCount': channelCount, 'FFT_Channel_Data': FFT_Channel_Data, 'imageCount': imageCount}

        return (tensors_out, FFT_Data)


def low_pass_filter(shape, cutoff):
    """生成低通滤波器，保留低于 cutoff 的频率"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    r, c = np.ogrid[:rows, :cols]
    distance = np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
    mask = (distance <= cutoff).astype(np.float32)
    return mask

def high_pass_filter(shape, cutoff):
    """生成高通滤波器，保留高于 cutoff 的频率"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    r, c = np.ogrid[:rows, :cols]
    distance = np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
    mask = (distance >= cutoff).astype(np.float32)
    return mask

def band_pass_filter(shape, low_cutoff, high_cutoff):
    """生成带通滤波器，保留 low_cutoff 到 high_cutoff 之间的频率"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    r, c = np.ogrid[:rows, :cols]
    distance = np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
    mask = np.logical_and(distance >= low_cutoff, distance <= high_cutoff).astype(np.float32)
    return mask
# 应用输入的自定义遮罩到傅里叶频谱上
def ApplyMask(l_fshift, l_mask, filter_type="low_pass", low_cutoff=10, high_cutoff=50):
    # 将 l_fshift 转换为 numpy 数组，确保兼容性
    if isinstance(l_fshift, torch.Tensor):
        l_fshift = l_fshift.cpu().numpy()
    
    # 将 mask 也转换为 numpy 数组
    if isinstance(l_mask, torch.Tensor):
        l_mask = l_mask.cpu().numpy()

    rows, cols = l_fshift.shape

    # 根据滤波器类型选择对应的滤波器
    if filter_type == "low_pass":
        filter_mask = low_pass_filter((rows, cols), high_cutoff)
    elif filter_type == "high_pass":
        filter_mask = high_pass_filter((rows, cols), low_cutoff)
    elif filter_type == "band_pass":
        filter_mask = band_pass_filter((rows, cols), low_cutoff, high_cutoff)

    # 结合输入的遮罩，滤波器只作用于指定区域
    combined_mask = l_mask * filter_mask

    # 应用滤波器，确保频谱和遮罩类型一致
    f = l_fshift * combined_mask  # 在频谱上应用滤波器，形状不变
    ishift = np.fft.ifftshift(f)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    
    return f, iimg



class LG_IFFTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ff": ("FFTData", ),
                "mask": ("MASK", ),  # 输入遮罩
                "filter_type": (["low_pass", "high_pass", "band_pass"], {"default": "low_pass"}),  # 滤波类型
                "low_cutoff": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "high_cutoff": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "invert_mask": ("BOOLEAN", {"default": False})  # 是否反转遮罩
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fromFFT"
    CATEGORY = "🎈LAOGOU"

    def DoOneChannel(self, fshift, mask, filter_type="low_pass", low_cutoff=10, high_cutoff=50, invert_mask=False):
        # 如果 invert_mask 为 True，则反转遮罩，采用 1 - mask 逻辑
        if invert_mask:
            mask = 1 - mask  # 将 1 变为 0，0 变为 1
        
        # 应用遮罩和滤波器
        fshift_masked, hi_pass_img = ApplyMask(fshift, mask, filter_type, low_cutoff, high_cutoff)
        return hi_pass_img / 255

    def fromFFT(self, ff, mask, filter_type="low_pass", low_cutoff=10, high_cutoff=50, invert_mask=False):
        channel_count = ff['channelCount']
        image_count = ff['imageCount']
        res = []

        for i in range(image_count):
            if channel_count == 3:
                # 分别处理 R、G、B 通道
                f0 = ff['FFT_Channel_Data'][i][0]
                f1 = ff['FFT_Channel_Data'][i][1]
                f2 = ff['FFT_Channel_Data'][i][2]

                # 对每个通道应用遮罩和滤波器，传递 invert_mask 参数
                out0 = self.DoOneChannel(f0, mask, filter_type, low_cutoff, high_cutoff, invert_mask)
                out1 = self.DoOneChannel(f1, mask, filter_type, low_cutoff, high_cutoff, invert_mask)
                out2 = self.DoOneChannel(f2, mask, filter_type, low_cutoff, high_cutoff, invert_mask)

                # 去掉多余的维度，使其变为 (931, 421)
                out0 = np.squeeze(out0)
                out1 = np.squeeze(out1)
                out2 = np.squeeze(out2)

                # 合并通道并确保形状为 (931, 421, 3)
                done_img = np.dstack((out0, out1, out2)).astype(np.float32)

                # 添加 batch 维度，确保其为 (1, 931, 421, 3)
                done_img = np.expand_dims(done_img, axis=0)

                # 添加结果到 res 列表中
                res.append(done_img)
            else:
                # 单通道图像处理
                f0 = ff['FFT_Channel_Data'][i]
                done_img = self.DoOneChannel(f0, mask, filter_type, low_cutoff, high_cutoff, invert_mask)

                # 去掉多余的维度，使其变为 (931, 421)
                done_img = np.squeeze(done_img.astype(np.float32))

                # 将单通道扩展为三通道 (复制三次以变为 RGB)
                done_img = np.stack([done_img] * 3, axis=-1)

                # 添加 batch 维度，确保其为 (1, 931, 421, 3)
                done_img = np.expand_dims(done_img, axis=0)

                # 添加结果到 res 列表中
                res.append(done_img)

        # 如果输出是单张图像，则堆叠结果并返回
        tensors_out = torch.stack([torch.from_numpy(np_array) for np_array in res])

        return tensors_out



NODE_CLASS_MAPPINGS = {
    "LG_FFTNode": LG_FFTNode,
    "LG_IFFTNode": LG_IFFTNode


}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LG_FFTNode": "🎈LG_FFT",
    "LG_IFFTNode": "🎈LG_IFFT"
}
