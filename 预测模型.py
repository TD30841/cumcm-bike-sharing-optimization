import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

TEST_DATE = '2025-04-13'
DATA_FILE = '附件2_每小时借还记录.csv'

record = pd.read_csv(DATA_FILE)

print("当前工作目录：", os.getcwd())
print("数据列名：", list(record.columns))
print("前5行数据：")
print(record.head())

record['日期'] = pd.to_datetime(record['日期'])

print("\n各列缺失值数量：")
print(record.isnull().sum())

print("\n站点数量：", record['站点编号'].nunique())
print("日期数量：", record['日期'].nunique())
print("总记录数：", len(record))

record['weekday'] = record['日期'].dt.weekday
record['日期类型'] = record['weekday'].apply(lambda x: '工作日' if x < 5 else '休息日')

test_date = pd.to_datetime(TEST_DATE)
train = record[record['日期'] < test_date].copy()
test = record[record['日期'] == test_date].copy()

print("\n训练集记录数：", len(train))
print("测试集记录数：", len(test))

if len(test) == 0:
    raise ValueError(f"测试日 {TEST_DATE} 没有数据，请检查日期是否写对。")

pred_m1 = (
    train.groupby(['站点编号', '小时(0-23)'])[['借出量', '归还量']]
    .mean()
    .reset_index()
    .rename(columns={
        '借出量': '模型1_预测借出量',
        '归还量': '模型1_预测归还量'
    })
)

result_m1 = test.merge(pred_m1, on=['站点编号', '小时(0-23)'], how='left')
result_m1['模型1_借出绝对误差'] = (result_m1['借出量'] - result_m1['模型1_预测借出量']).abs()
result_m1['模型1_归还绝对误差'] = (result_m1['归还量'] - result_m1['模型1_预测归还量']).abs()

mae_m1_borrow = result_m1['模型1_借出绝对误差'].mean()
mae_m1_return = result_m1['模型1_归还绝对误差'].mean()

pred_m2 = (
    train.groupby(['站点编号', '小时(0-23)', '日期类型'])[['借出量', '归还量']]
    .mean()
    .reset_index()
    .rename(columns={
        '借出量': '模型2_预测借出量',
        '归还量': '模型2_预测归还量'
    })
)

final_pred_workday = (
    pred_m2[pred_m2['日期类型'] == '工作日'][['站点编号', '小时(0-23)', '日期类型', '模型2_预测借出量', '模型2_预测归还量']]
    .copy()
    .rename(columns={
        '模型2_预测借出量': '预测借出量',
        '模型2_预测归还量': '预测归还量'
    })
)

final_pred_workday.to_csv('最终预测结果_模型2_工作日.csv', index=False, encoding='utf-8-sig')
print("已导出：最终预测结果_模型2_工作日.csv")
print("工作日预测表记录数：", len(final_pred_workday))
print("工作日预测借出量总和：", final_pred_workday['预测借出量'].sum())
print("工作日预测归还量总和：", final_pred_workday['预测归还量'].sum())

result_m2 = test.merge(
    pred_m2,
    on=['站点编号', '小时(0-23)', '日期类型'],
    how='left'
)

result_m2['模型2_借出绝对误差'] = (result_m2['借出量'] - result_m2['模型2_预测借出量']).abs()
result_m2['模型2_归还绝对误差'] = (result_m2['归还量'] - result_m2['模型2_预测归还量']).abs()

mae_m2_borrow = result_m2['模型2_借出绝对误差'].mean()
mae_m2_return = result_m2['模型2_归还绝对误差'].mean()

summary = pd.DataFrame({
    '模型': ['模型1_历史同期均值法', '模型2_区分工作日休息日均值法'],
    '借出量_MAE': [mae_m1_borrow, mae_m2_borrow],
    '归还量_MAE': [mae_m1_return, mae_m2_return]
})

print("\n=== 模型比较结果 ===")
print(summary)

better_borrow = '模型1' if mae_m1_borrow < mae_m2_borrow else '模型2'
better_return = '模型1' if mae_m1_return < mae_m2_return else '模型2'

print(f"\n借出量预测更优：{better_borrow}")
print(f"归还量预测更优：{better_return}")

summary.to_csv('模型比较结果_MAE.csv', index=False, encoding='utf-8-sig')
result_m1.to_csv('模型1_预测结果_历史同期均值法.csv', index=False, encoding='utf-8-sig')
result_m2.to_csv('模型2_预测结果_区分工作日休息日.csv', index=False, encoding='utf-8-sig')

hour_plot_m1 = (
    result_m1.groupby('小时(0-23)')[['借出量', '模型1_预测借出量', '归还量', '模型1_预测归还量']]
    .sum()
    .reset_index()
)

hour_plot_m2 = (
    result_m2.groupby('小时(0-23)')[['借出量', '模型2_预测借出量', '归还量', '模型2_预测归还量']]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(hour_plot_m1['小时(0-23)'], hour_plot_m1['借出量'], marker='o', label='实际借出量')
plt.plot(hour_plot_m1['小时(0-23)'], hour_plot_m1['模型1_预测借出量'], marker='s', label='模型1预测借出量')
plt.xlabel('小时')
plt.ylabel('总借出量')
plt.title(f'模型1：{TEST_DATE} 各小时总借出量预测对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('图_模型1借出量预测对比.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(hour_plot_m2['小时(0-23)'], hour_plot_m2['借出量'], marker='o', label='实际借出量')
plt.plot(hour_plot_m2['小时(0-23)'], hour_plot_m2['模型2_预测借出量'], marker='s', label='模型2预测借出量')
plt.xlabel('小时')
plt.ylabel('总借出量')
plt.title(f'模型2：{TEST_DATE} 各小时总借出量预测对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('图_模型2借出量预测对比.png', dpi=300, bbox_inches='tight')
plt.close()

station_error_m1 = (
    result_m1.groupby('站点编号')[['模型1_借出绝对误差', '模型1_归还绝对误差']]
    .mean()
    .reset_index()
    .sort_values('模型1_借出绝对误差', ascending=False)
)

station_error_m2 = (
    result_m2.groupby('站点编号')[['模型2_借出绝对误差', '模型2_归还绝对误差']]
    .mean()
    .reset_index()
    .sort_values('模型2_借出绝对误差', ascending=False)
)

station_error_m1.to_csv('模型1_各站点平均误差.csv', index=False, encoding='utf-8-sig')
station_error_m2.to_csv('模型2_各站点平均误差.csv', index=False, encoding='utf-8-sig')

print("\n文件已生成：")
print("1. 模型比较结果_MAE.csv")
print("2. 模型1_预测结果_历史同期均值法.csv")
print("3. 模型2_预测结果_区分工作日休息日.csv")
print("4. 图_模型1借出量预测对比.png")
print("5. 图_模型2借出量预测对比.png")
print("6. 模型1_各站点平均误差.csv")
print("7. 模型2_各站点平均误差.csv")

summary_plot = summary.copy()
x = range(len(summary_plot))

plt.figure(figsize=(8, 6))
plt.bar([i - 0.15 for i in x], summary_plot['借出量_MAE'], width=0.3, label='借出量 MAE')
plt.bar([i + 0.15 for i in x], summary_plot['归还量_MAE'], width=0.3, label='归还量 MAE')
plt.xticks(list(x), summary_plot['模型'], rotation=0)
plt.ylabel('MAE')
plt.title('不同预测模型的 MAE 对比')
plt.legend()
plt.tight_layout()
plt.savefig('图_模型MAE对比柱状图.png', dpi=300, bbox_inches='tight')
plt.show()
