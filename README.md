# 基于clip的开放词汇语义分割模型结构对比


本项目对比分析了三个基于 CLIP 的开放词汇语义分割模型：

1. [**OVSeg**](https://github.com/facebookresearch/ov-seg)：通过训练视觉 prompt 适配 CLIP 到被遮挡图像上  
2. [**Side Adapter Network (SAN)**](https://github.com/MendelXu/SAN)：在不修改 CLIP 主干的基础上，引入旁路结构提供空间感知能力  
3. [**CLIP as RNN (CaR)**](https://github.com/google-research/google-research/tree/master/clip_as_rnn)：使用视觉提示 + 递归推理方式，引导 CLIP 逐步完成分割任务

我们对这三种方法的结构差异与推理流程进行了分析，并修改了部分由于包版本更新导致的问题，便于运行测试与对比。



---

## 🚀 模型使用说明

### 1️⃣ OVSeg 模型运行

- **运行 Demo 示例**
```bash
cd ov-seg-main
python demo.py \
  --config-file configs/ovseg_swinB_vitL_demo.yaml \
  --class-names 'Oculus' 'Ukulele' \
  --input ./resources/demo_samples/sample_03.jpeg \
  --output ./pred \
  --opts MODEL.WEIGHTS path/to/your/weight.pth
```

---

### 2️⃣ SAN 模型运行

- **安装依赖**
```bash
git clone https://github.com/MendelXu/SAN.git
cd SAN
bash install.sh
```

- **使用 Docker 运行 Demo**
```bash
docker build docker/app.Docker -t san_app
docker run -it --shm-size 4G -p 7860:7860 san_app
```

- **训练模型**
```bash
cd SAN
python train_net.py \
  --config-file configs/san_clip_vit_res4_coco.yaml \
  --num-gpus 1 \
  OUTPUT_DIR ./output/san_vit_b16
```

- **评估模型**
```bash
python train_net.py --eval-only \
  --config-file configs/san_clip_vit_res4_coco.yaml \
  --num-gpus 1 \
  OUTPUT_DIR ./output/san_vit_b16_eval \
  MODEL.WEIGHTS output/san_vit_b_16.pth \
  DATASETS.TEST "('ade20k_sem_seg_val',)"
```

- **可视化分割结果**
```bash
python visualize_json_results.py \
  --input output/san_vit_b16_eval/inference/sem_seg_predictions.json \
  --output output/san_vit_b16_eval/viz \
  --dataset ade20k_sem_seg_val
```

---

### 3️⃣ CLIP as RNN 模型运行

- **运行单张图像测试**
```bash
cd clip_as_rnn
python3 demo.py \
  --cfg-path=configs/ade20k/car_vitb_clip.yaml \
  --output_path=results/demo
```

- **在数据集上评估模型**
```bash
python3 evaluate.py --cfg-path=configs/ade20k/car_vitb_clip.yaml
```

---

## 📚 引用信息Citation


```bibtex
@inproceedings{clip_as_rnn,
  title = {CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor},
  author = {Sun, Shuyang and Li, Runjia and Torr, Philip and Gu, Xiuye and Li, Siyang},
  year = {2024},
  booktitle = {CVPR},
}

@article{xu2023san,
  title={SAN: Side adapter network for open-vocabulary semantic segmentation},
  author={Xu, Mengde and Zhang, Zheng and Wei, Fangyun and Hu, Han and Bai, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}

@inproceedings{xu2023side,
  title={Side Adapter Network for Open-Vocabulary Semantic Segmentation},
  author={Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, Xiang Bai},
  journal={CVPR},
  year={2023}
}

@inproceedings{liang2023open,
  title={Open-vocabulary semantic segmentation with mask-adapted clip},
  author={Liang, Feng and Wu, Bichen and Dai, Xiaoliang and Li, Kunpeng and Zhao, Yinan and Zhang, Hang and Zhang, Peizhao and Vajda, Peter and Marculescu, Diana},
  booktitle={CVPR},
  pages={7061--7070},
  year={2023}
}
```
