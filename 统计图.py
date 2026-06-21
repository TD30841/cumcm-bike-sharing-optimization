import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

site = pd.read_csv('附件1_站点基础信息.csv')
record = pd.read_csv('附件2_每小时借还记录.csv')

record['日期'] = pd.to_datetime(record['日期'])
record['weekday'] = record['日期'].dt.weekday
record['日期类型'] = record['weekday'].apply(lambda x: '工作日' if x < 5 else '休息日')

station_borrow = record.groupby('站点编号', as_index=False)['借出量'].sum()
station_borrow['日均借出量'] = station_borrow['借出量'] / record['日期'].nunique()
station_borrow = station_borrow.merge(
    site[['站点编号', '站点名称', '经度', '纬度']],
    on='站点编号',
    how='left'
)
station_borrow = station_borrow.sort_values('日均借出量', ascending=False).reset_index(drop=True)

group_labels = ['高借出量组', '中高借出量组', '中等偏高组', '中等偏低组', '中低借出量组', '低借出量组']
station_borrow['组序号'] = station_borrow.index // 5
station_borrow['组别'] = station_borrow['组序号'].map(dict(enumerate(group_labels)))

top5 = station_borrow.head(5).copy()
bottom5 = station_borrow.tail(5).copy()
plot_bar = pd.concat([top5, bottom5], axis=0)

color_map = {
    '高借出量组': '#08306b',
    '中高借出量组': '#2171b5',
    '中等偏高组': '#4292c6',
    '中等偏低组': '#6baed6',
    '中低借出量组': '#9ecae1',
    '低借出量组': '#c6dbef'
}

plt.figure(figsize=(12, 6))
plt.bar(
    plot_bar['站点名称'],
    plot_bar['日均借出量'],
    color=plot_bar['组别'].map(color_map)
)
plt.xlabel('站点名称')
plt.ylabel('日均借出量')
plt.title('Top5 / Bottom5 站点日均借出量柱状图')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('图1_Top5_Bottom5站点日均借出量柱状图.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

size_map = {
    '高借出量组': 260,
    '中高借出量组': 220,
    '中等偏高组': 180,
    '中等偏低组': 140,
    '中低借出量组': 100,
    '低借出量组': 60
}

plt.figure(figsize=(9, 8))
for g in group_labels:
    temp = station_borrow[station_borrow['组别'] == g]
    plt.scatter(
        temp['经度'],
        temp['纬度'],
        s=temp['组别'].map(size_map),
        c=color_map[g],
        alpha=0.8,
        label=g,
        edgecolors='black'
    )

for _, row in top5.iterrows():
    plt.annotate(
        row['站点名称'],
        (row['经度'], row['纬度']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        color='darkred'
    )

for _, row in bottom5.iterrows():
    plt.annotate(
        row['站点名称'],
        (row['经度'], row['纬度']),
        xytext=(5, -10),
        textcoords='offset points',
        fontsize=9,
        color='darkgreen'
    )

plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('站点空间分布散点图（按日均借出量分组）')
plt.legend(title='组别')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('图2_站点空间分布散点图.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

station_borrow.to_csv('站点日均借出量及分组结果.csv', index=False, encoding='utf-8-sig')

# 第二个模型：按 站点 + 小时 + 日期类型 求历史均值
latest_date = record['日期'].max()
train = record[record['日期'] < latest_date].copy()
test = record[record['日期'] == latest_date].copy()

pred_m2 = (
    train.groupby(['站点编号', '小时(0-23)', '日期类型'])[['借出量', '归还量']]
    .mean()
    .reset_index()
    .rename(columns={
        '借出量': '预测借出量',
        '归还量': '预测归还量'
    })
)

result_m2 = test.merge(
    pred_m2,
    on=['站点编号', '小时(0-23)', '日期类型'],
    how='left'
)

result_m2 = result_m2.merge(
    station_borrow[['站点编号', '站点名称', '组别', '日均借出量']],
    on='站点编号',
    how='left'
)

# 选三个代表性站点：高借出量、中等借出量、低借出量
rep_high = station_borrow.iloc[0]
rep_mid = station_borrow.iloc[len(station_borrow) // 2]
rep_low = station_borrow.iloc[-1]

rep_stations = [
    ('高借出量代表站点', rep_high['站点编号'], rep_high['站点名称']),
    ('中等借出量代表站点', rep_mid['站点编号'], rep_mid['站点名称']),
    ('低借出量代表站点', rep_low['站点编号'], rep_low['站点名称'])
]

summary_rows = []

for label, sid, sname in rep_stations:
    temp = result_m2[result_m2['站点编号'] == sid].copy().sort_values('小时(0-23)')

    temp['借出误差'] = temp['预测借出量'] - temp['借出量']
    temp['归还误差'] = temp['预测归还量'] - temp['归还量']

    summary_rows.append({
        '代表类型': label,
        '站点编号': sid,
        '站点名称': sname,
        '组别': temp['组别'].iloc[0] if len(temp) > 0 else '',
        '借出量MAE': (temp['借出误差'].abs().mean() if len(temp) > 0 else None),
        '归还量MAE': (temp['归还误差'].abs().mean() if len(temp) > 0 else None),
        '借出量实际总量': temp['借出量'].sum(),
        '借出量预测总量': temp['预测借出量'].sum(),
        '归还量实际总量': temp['归还量'].sum(),
        '归还量预测总量': temp['预测归还量'].sum(),
        '测试日期': latest_date.strftime('%Y-%m-%d')
    })

    plt.figure(figsize=(10, 6))
    plt.plot(temp['小时(0-23)'], temp['借出量'], marker='o', linewidth=2, label='实际借出量')
    plt.plot(temp['小时(0-23)'], temp['预测借出量'], marker='s', linewidth=2, label='预测借出量')
    plt.plot(temp['小时(0-23)'], temp['归还量'], marker='^', linewidth=2, linestyle='--', label='实际归还量')
    plt.plot(temp['小时(0-23)'], temp['预测归还量'], marker='d', linewidth=2, linestyle='--', label='预测归还量')
    plt.xlabel('小时')
    plt.ylabel('数量')
    plt.title(f'{label}（{sname}，{sid}）借出归还量实际值与预测值对比')
    plt.xticks(range(0, 24, 1))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'图3_{sid}_{sname}_四线对比图.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('三个代表性站点预测效果汇总.csv', index=False, encoding='utf-8-sig')
result_m2.to_csv('模型2_测试日详细结果.csv', index=False, encoding='utf-8-sig')

print('图表和结果表已生成完成。')
print('文件包括：')
print('1. 图1_Top5_Bottom5站点日均借出量柱状图.png')
print('2. 图2_站点空间分布散点图.png')
print('3. 站点日均借出量及分组结果.csv')
print('4. 图3_三个代表性站点四线对比图（共3张）')
print('5. 三个代表性站点预测效果汇总.csv')
print('6. 模型2_测试日详细结果.csv')
