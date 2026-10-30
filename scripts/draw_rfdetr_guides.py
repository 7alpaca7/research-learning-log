"""Draw explanatory figures for the RF-DETR reading note."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rfdetr-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'assets/rfdetr'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
INK='#21364d';BLUE='#e6f0fc';GREEN='#e3f3ed';ORANGE='#fff0df'
def t(ax,x,y,s,size=12,color=INK):ax.text(x,y,s,ha='center',va='center',fontproperties=FONT,fontsize=size,color=color)
def box(ax,x,y,w,h,s,c=BLUE,size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.1',fc=c,ec='#b4c7d7'));t(ax,x+w/2,y+h/2,s,size)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.7,color='#67869e',shrinkA=4,shrinkB=4))
def canvas(title,sub,h=6):
 fig,ax=plt.subplots(figsize=(12,h));ax.set(xlim=(0,12),ylim=(0,h));ax.axis('off');t(ax,6,h-.4,title,19);t(ax,6,h-.9,sub,10);return fig,ax
def save(fig,name):fig.savefig(OUT/name,dpi=190,bbox_inches='tight',pad_inches=.18);plt.close(fig)
# 1: normalization
fig,ax=canvas('梯度累积不会合并 BN 的前向统计量','示意：有效 batch = 16，微批次 = 4，累积 4 次后更新参数',6)
for i in range(4):
 x=.35+i*2.95;box(ax,x,3.35,2.5,1,'微批次 '+str(i+1)+'\n4 张图像',BLUE)
 box(ax,x,1.9,2.5,.85,'梯度 g'+str(i+1),GREEN);arrow(ax,(x+1.25,3.35),(x+1.25,2.75))
t(ax,6,1.25,'按损失缩放约定累积梯度 → 一次 optimizer.step()',13)
box(ax,.45,.15,5.3,.6,'BN：每次只统计当前微批次（及空间维度）',ORANGE,11)
box(ax,6.25,.15,5.3,.6,'LN：在每个样本指定的特征维度内统计',GREEN,11)
save(fig,'guide-normalization.png')
# 2 masks
fig,ax=canvas('分割头：目标 Query 与像素嵌入在通道维度对齐','简化张量示意；展示单个 Decoder 层，省略逐层细化和训练损失',6)
box(ax,.3,2.25,2,1.2,'共享低分辨率\n空间特征',BLUE)
box(ax,3.1,3.15,2.7,1.0,'上采样 + 轻量投影\n像素嵌入 E：Hm × Wm × d',GREEN,11)
box(ax,3.1,1.25,2.7,1.0,'Decoder Query → FFN\nQ：K × d',ORANGE,11)
arrow(ax,(2.3,2.85),(3.1,3.65));box(ax,6.65,2.25,2.0,1.2,'通道点积\nQ × transpose(E)',BLUE)
arrow(ax,(5.8,3.65),(6.65,3));arrow(ax,(5.8,1.75),(6.65,2.65))
box(ax,9.45,2.25,2.2,1.2,'K 张 Mask logits\nK × Hm × Wm',GREEN,11);arrow(ax,(8.65,2.85),(9.45,2.85))
t(ax,6,.55,'每个 Query 为同一张像素嵌入图提供一组匹配权重，得到对应目标的空间响应。',12)
save(fig,'guide-mask-head.png')
# 3 grid
fig,ax=canvas('分辨率和 Patch Size：一起看 Token 网格','假设正方形输入且尺寸可整除；N = (H / P)²，不含额外 Token',6.3)
for x,H,P,n,c in [(.7,640,16,40,BLUE),(4.6,320,8,40,GREEN),(8.5,640,32,20,ORANGE)]:
 # Draw a 10x10 or 5x5 representative coarse grid, explicitly schematic
 k=10 if n==40 else 5
 for i in range(k):
  for j in range(k):ax.add_patch(Rectangle((x+i*2.7/k,1.85+j*2.7/k),2.7/k,2.7/k,fc=c,ec='#92afc6',lw=.55))
 t(ax,x+1.35,4.85,f'H = {H}，P = {P}',13)
 t(ax,x+1.35,1.4,f'{n} × {n} = {n*n} Tokens',12)
t(ax,6,.7,'左、中两组 Token 数相同，但输入缩放造成的信息损失和实际延迟仍可能不同。',11)
t(ax,6,.25,'网格为疏化示意，不按真实 Token 个数逐格绘制；Token 数不等于完整模型 FLOPs。',10)
save(fig,'guide-token-grid.png')
# 4 decoder
fig,ax=canvas('两个独立的推理开关：Decoder 深度与 Query 数量','训练：固定 6 层 Decoder、300 Queries；推理：截断层数并按 Encoder 评分筛选候选',6.5)
t(ax,1.1,4.7,'训练监督',12)
for i in range(6):
 x=2+i*1.5;box(ax,x,4.25,1.15,.85,'Layer '+str(i+1),BLUE,11);t(ax,x+.575,3.9,'↓ loss',10)
 if i<5:arrow(ax,(x+1.15,4.68),(x+1.5,4.68))
t(ax,1.1,3.1,'截断示例',12)
for i in range(6):
 x=2+i*1.5;box(ax,x,2.72,1.15,.65,str(i+1),GREEN if i<3 else '#f1f2f4',11)
t(ax,5,2.2,'执行前 3 层 → 使用该层预测',11);t(ax,9.7,2.2,'后三层不执行',11)
box(ax,.6,.65,3.15,.85,'Encoder 候选\n按最大类别 sigmoid 分数排序',ORANGE,11)
box(ax,4.45,.65,2.8,.85,'Top-K 筛选\n例如 K = 100',GREEN,11)
box(ax,8,.65,3.35,.85,'送入选定深度的 Decoder\n减少候选处理开销',BLUE,11)
arrow(ax,(3.75,1.07),(4.45,1.07));arrow(ax,(7.25,1.07),(8,1.07))
t(ax,6,.15,'筛选候选不等于 NMS；低分真实目标被提前删除的风险需要用召回率验证。',10)
save(fig,'guide-decoder-query.png')
# 5 window attention
fig,ax=canvas('局部窗口减少交互范围，全局层补充跨窗口联系','8 × 8 Token 网格的概念示意；颜色表示局部注意力窗口，不对应具体型号配置',6.2)
colors=['#d9e8fa','#d9efe5','#f9e5ce','#e7def5']
for j in range(8):
 for i in range(8):
  c=colors[(j//4)*2+i//4];ax.add_patch(Rectangle((.8+i*.36,1.5+j*.36),.36,.36,fc=c,ec='white'))
box(ax,5.1,2.35,2.3,1.15,'Non-windowed\n全局注意力层',GREEN,13)
arrow(ax,(3.8,2.95),(5.1,2.95));arrow(ax,(7.4,2.95),(8.5,2.95))
box(ax,8.6,2.15,2.8,1.55,'跨窗口信息交互\n\n远距离 Token 建立联系',BLUE,12)
t(ax,2.25,1.05,'Windowed 层：窗口内交互',11)
t(ax,6,.5,'固定 N 个 Token、等分为 W 个窗口时，局部注意力项约为 N² / W。',12)
t(ax,6,.08,'该式忽略投影、FFN、全局层和硬件开销；不能直接推导完整模型的加速倍数。',10)
save(fig,'guide-window-attention.png')
# 6 padding
fig,ax=canvas('Batch-Level Resize：减少尺寸不一致带来的填充','图示仅解释逐图随机尺寸与批次统一目标尺寸的区别；不代表精确预处理实现',6.1)
for base,title,shapes in [(.5,'逐图随机 Resize',[(1.2,1.2),(1.8,1.8),(1.5,1.5)]),(6.6,'批次统一 Resize 配置',[(1.8,1.8)]*3)]:
 t(ax,base+2.35,4.55,title,15)
 for j,(w,h) in enumerate(shapes):
  x=base+j*1.75
  ax.add_patch(Rectangle((x,2.15),1.5,1.8,fc='#edf0f4',ec='#a8b8c8'))
  ax.add_patch(Rectangle((x,2.15),w/1.8*1.5,h/1.8*1.8,fc='#96c7d4',ec='white'))
 t(ax,base+2.35,1.7,'蓝色：图像区域    灰色：Padding',10)
t(ax,6,.9,'目标：少计算填充区域，缓解窗口相关伪影，让分辨率采样更均衡。',12)
t(ax,6,.35,'示意采用相同长宽比图像；真实数据仍可能因长宽比、裁剪和整除约束而需要 Padding。',10)
save(fig,'guide-batch-resize.png')
