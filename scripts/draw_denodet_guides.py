"""Regenerate original explanatory figures for the DenoDet reading note."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/denodet-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
OUT=Path(__file__).resolve().parents[1]/'assets/denodet'
INK='#23364d'; BLUE='#e9f2ff'; GREEN='#e8f5ef'; ORANGE='#fff0df'
def txt(ax,x,y,s,size=12,**kw):
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,color=INK,ha='center',va='center',**kw)
def box(ax,x,y,w,h,s,c=BLUE,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.12',facecolor=c,edgecolor='#b8c9db',lw=1.1))
    txt(ax,x+w/2,y+h/2,s,size)
def arrow(ax,a,b,c='#738ca7',style='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=c,lw=1.7,linestyle=style,shrinkA=3,shrinkB=3))
def canvas(title,sub,h=5):
    fig,ax=plt.subplots(figsize=(12,h));fig.patch.set_facecolor('white')
    ax.set(xlim=(0,12),ylim=(0,h));ax.axis('off')
    txt(ax,6,h-.35,title,19,weight='bold');txt(ax,6,h-.83,sub,10)
    return fig,ax
def save(fig,name):
    fig.savefig(OUT/name,dpi=180,bbox_inches='tight',pad_inches=.16);plt.close(fig)

fig,ax=canvas('三层设计：每一步回应一个新的问题','自绘阅读图｜从算子解释，到表示选择，再到交互结构',5)
for x,c,n,t,b in [(0.3,BLUE,'01','注意力为什么像去噪？','门控与动态软阈值\n建立条件等价'),(4.3,GREEN,'02','在什么表示下收缩？','DCT → 系数调节 → IDCT\n引入固定的频率坐标'),(8.3,ORANGE,'03','如何选择交互范围？','分组 FC → 动态融合\n适配不同输入的交互粒度')]:
    box(ax,x,1.45,3.35,2.0,n+'  '+t+'\n\n'+b,c,12)
arrow(ax,(3.7,2.45),(4.25,2.45));arrow(ax,(7.7,2.45),(8.25,2.45))
txt(ax,6,.65,'收缩是数学描述；被抑制的响应是否属于噪声，仍需要额外证据。',11)
save(fig,'guide-design-logic.png')

fig,ax=canvas('TransDeno：特征主路与权重支路分开看','自绘计算示意｜按软阈值与门控等价成立的情形简化；省略转置和重排',6)
for x,w,t,c in [(.15,1.5,'空间特征\nC × H × W',BLUE),(2.15,1.5,'DCT 系数 m\nC × H × W',BLUE),(7.65,1.8,'逐频率门控\na ⊙ m',GREEN),(10.05,1.65,'IDCT\n空间特征',BLUE)]:
    box(ax,x,3.35,w,1,t,c)
for a,b in [((1.65,3.85),(2.15,3.85)),((3.65,3.85),(7.65,3.85)),((9.45,3.85),(10.05,3.85))]:arrow(ax,a,b)
txt(ax,5.6,4.12,'保留频域系数的主路',11)
box(ax,2.05,1.05,2.4,1.1,'沿通道 Avg + Max\n展平为 N = HW',ORANGE)
box(ax,5.05,1.05,3.25,1.1,'DeGroFC → ReLU\n→ DeGroFC → Sigmoid',ORANGE)
box(ax,8.95,1.05,2.5,1.1,'频率权重 a\n沿通道广播',GREEN)
arrow(ax,(2.9,3.35),(3.25,2.15));arrow(ax,(4.45,1.6),(5.05,1.6));arrow(ax,(8.3,1.6),(8.95,1.6));arrow(ax,(10.2,2.15),(8.55,3.35),c='#39947b')
txt(ax,6,.35,'统计分支先压缩通道；主路保留各通道系数，再使用频率权重调节。',11)
save(fig,'guide-two-paths.png')

fig,ax=canvas('分组改变交互范围，连续配置组合候选映射','自绘概念图｜左侧取 N = 16；右侧仅展示一个配置值的常规线性插值参照',6.3)
colors=['#568ec7','#58a994','#e7ad64','#a891c5','#d88a90','#6ba6af','#9aad68','#b69478']
for y,g in [(4.1,2),(3.2,4),(2.3,8)]:
    txt(ax,.65,y+.22,'g = '+str(g),12)
    for j in range(16):
        x=1.3+j*.265
        ax.add_patch(Rectangle((x,y),.23,.43,facecolor=colors[j//(16//g)],edgecolor='white'))
    txt(ax,3.4,y-.2,'每组 '+str(16//g)+' 个输入位置；组内映射',10)
box(ax,6.35,3.75,2.05,.8,'候选索引 1\n分组数 4',BLUE)
box(ax,9.2,3.75,2.05,.8,'候选索引 2\n分组数 8',GREEN)
box(ax,7.8,2.15,2.05,.7,'配置值 K_i = 1.3',ORANGE)
arrow(ax,(8.0,2.85),(7.4,3.75));arrow(ax,(9.6,2.85),(10.25,3.75))
txt(ax,8.75,1.65,'参考输出：0.7 G_4(s) + 0.3 G_8(s)',12)
txt(ax,6,.75,'颜色只表示分组边界，不代表目标 / 背景，也不假定展平后就是理想频带。',10)
txt(ax,6,.32,'右图不是原文公式的复写：原文还有多个 K_i 求和及 Softmax，权重和边界需核查。',10)
save(fig,'guide-grouping.png')
