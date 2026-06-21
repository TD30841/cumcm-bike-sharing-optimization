import pandas as pd
import numpy as np
import math
import os
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpStatus,
    HiGHS, value, LpInteger, LpContinuous
)

SITE_FILE = '附件1_站点基础信息.csv'
PRED_FILE = '最终预测结果_模型2_工作日.csv'

TARGET_DAY_TYPE = '工作日'
HOURS = list(range(24))

SAFETY_RATIO = 0.20
TRUCK_CAPACITY = 20
SPEED_KMPH = 12
TRANSPORT_COST_MODE = 'per_trip'
COST_PER_KM = 2.0
FULL_PENALTY = 15.0
EMPTY_PENALTY = 10.0

SERVICE_WEIGHT = 0.0

SOLVER_TIME_LIMIT = 300

def find_col(df, candidates, required=True):

    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'没有找到列名，候选列表是：{candidates}\n当前表列名为：{list(df.columns)}')
    return None

def haversine_km(lon1, lat1, lon2, lat2):

    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def safe_round_travel_time(distance_km, speed_kmph):

    raw = distance_km / speed_kmph
    return max(1, int(round(raw)))

site = pd.read_csv(SITE_FILE)
pred = pd.read_csv(PRED_FILE)

print("当前工作目录：", os.getcwd())
print("站点表列名：", list(site.columns))
print("预测表列名：", list(pred.columns))

site_id_col = find_col(site, ['站点编号', 'site_id', '站点ID'])
lon_col = find_col(site, ['经度', 'lon', 'longitude'])
lat_col = find_col(site, ['纬度', 'lat', 'latitude'])

cap_col = find_col(site, ['容量上限', '站点容量', '总桩位数', '容量', 'cap'])

init_col = find_col(site, ['当前库存量', '初始库存', '初始车辆数', '初始车数', '初始单车数', 'inventory0'], required=False)

safe_col = find_col(site, ['安全库存', '安全库存下限', 'safety_stock'], required=False)
name_col = find_col(site, ['站点名称', 'name'], required=False)

site_std = pd.DataFrame({
    'site_id': site[site_id_col],
    'lon': site[lon_col].astype(float),
    'lat': site[lat_col].astype(float),
    'capacity': site[cap_col].astype(float)
})

if name_col is not None:
    site_std['site_name'] = site[name_col]
else:
    site_std['site_name'] = site_std['site_id'].astype(str)

if init_col is not None:
    site_std['init_inventory'] = site[init_col].astype(float)
else:
    site_std['init_inventory'] = site_std['capacity'] / 2
    print("初始库存列设为容量的一半。")

if safe_col is not None:
    site_std['safe_stock'] = site[safe_col].astype(float)
else:
    site_std['safe_stock'] = SAFETY_RATIO * site_std['capacity']
    print(f"安全库存列，已默认设为容量的 {SAFETY_RATIO:.0%}。")

site_std = site_std.sort_values('site_id').reset_index(drop=True)
site_ids = site_std['site_id'].tolist()

full_penalty = {sid: FULL_PENALTY for sid in site_ids}
empty_penalty = {sid: EMPTY_PENALTY for sid in site_ids}

capacity = dict(zip(site_std['site_id'], site_std['capacity']))
init_inventory = dict(zip(site_std['site_id'], site_std['init_inventory']))
safe_stock = dict(zip(site_std['site_id'], site_std['safe_stock']))

pred_site_col = find_col(pred, ['站点编号', 'site_id', '站点ID'])
pred_hour_col = find_col(pred, ['小时(0-23)', '小时', 'hour'])
pred_type_col = find_col(pred, ['日期类型', 'day_type'], required=False)
pred_b_col = find_col(pred, ['预测借出量', '模型2_预测借出量'])
pred_r_col = find_col(pred, ['预测归还量', '模型2_预测归还量'])

pred_std = pd.DataFrame({
    'site_id': pred[pred_site_col],
    'hour': pred[pred_hour_col].astype(int),
    'pred_borrow': pred[pred_b_col].astype(float),
    'pred_return': pred[pred_r_col].astype(float)
})

if pred_type_col is not None:
    pred_std['day_type'] = pred[pred_type_col]
    pred_std = pred_std[pred_std['day_type'] == TARGET_DAY_TYPE].copy()


pred_std = pred_std.groupby(['site_id', 'hour'], as_index=False)[['pred_borrow', 'pred_return']].mean()

full_index = pd.MultiIndex.from_product([site_ids, HOURS], names=['site_id', 'hour'])
pred_std = (
    pred_std.set_index(['site_id', 'hour'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

B = {(row['site_id'], int(row['hour'])): float(row['pred_borrow']) for _, row in pred_std.iterrows()}
R = {(row['site_id'], int(row['hour'])): float(row['pred_return']) for _, row in pred_std.iterrows()}

dist = {}
tau = {}

coord_map = {row['site_id']: (row['lon'], row['lat']) for _, row in site_std.iterrows()}

for i in site_ids:
    lon1, lat1 = coord_map[i]
    for j in site_ids:
        if i == j:
            continue
        lon2, lat2 = coord_map[j]
        d = haversine_km(lon1, lat1, lon2, lat2)

        if i != j and d < 0.3:
            d = 0.3

        dist[(i, j)] = d
        tau[(i, j)] = safe_round_travel_time(d, SPEED_KMPH)


def simulate_no_dispatch(site_ids, hours, B, R, capacity, init_inventory, safe_stock,
                         full_penalty, empty_penalty):

    inv = {sid: float(init_inventory[sid]) for sid in site_ids}
    rows = []

    total_transport_cost = 0.0
    total_full_penalty = 0.0
    total_empty_penalty = 0.0
    full_occ = 0
    empty_occ = 0
    borrow_fail_count = 0.0
    return_fail_count = 0.0
    total_demand = 0.0

    for t in hours:
        for sid in site_ids:
            start_inv = inv[sid]
            need_b = B[(sid, t)]
            need_r = R[(sid, t)]
            total_demand += need_b + need_r

            success_b = min(start_inv, need_b)
            fail_b = need_b - success_b

            after_borrow = start_inv - success_b

            free_slots = capacity[sid] - after_borrow
            success_r = min(free_slots, need_r)
            fail_r = need_r - success_r

            end_inv = after_borrow + success_r

            low_gap = max(safe_stock[sid] - end_inv, 0)

            cost_full = full_penalty[sid] * fail_r
            cost_empty = empty_penalty[sid] * low_gap

            total_full_penalty += cost_full
            total_empty_penalty += cost_empty
            borrow_fail_count += fail_b
            return_fail_count += fail_r

            if fail_r > 1e-8:
                full_occ += 1
            if low_gap > 1e-8:
                empty_occ += 1

            rows.append({
                'site_id': sid,
                'hour': t,
                'start_inventory': start_inv,
                'pred_borrow': need_b,
                'pred_return': need_r,
                'success_borrow': success_b,
                'success_return': success_r,
                'borrow_fail': fail_b,
                'return_fail': fail_r,
                'end_inventory': end_inv,
                'low_gap': low_gap,
                'transport_cost': 0.0,
                'full_penalty_cost': cost_full,
                'empty_penalty_cost': cost_empty,
                'total_cost': cost_full + cost_empty
            })

            inv[sid] = end_inv

    total_cost = total_transport_cost + total_full_penalty + total_empty_penalty
    success_total = total_demand - borrow_fail_count - return_fail_count
    demand_rate = 100 * success_total / total_demand if total_demand > 0 else 0

    summary = {
        '方案': '无调度',
        '运输成本': total_transport_cost,
        '满桩惩罚成本': total_full_penalty,
        '空桩惩罚成本': total_empty_penalty,
        '日总费用': total_cost,
        '满桩发生次数': full_occ,
        '空桩发生次数': empty_occ,
        '用户无法借车次数': borrow_fail_count,
        '用户无法还车次数': return_fail_count,
        '用户需求率(%)': demand_rate
    }

    detail_df = pd.DataFrame(rows)
    return summary, detail_df

def optimize_dispatch_milp(site_ids, hours, B, R, capacity, init_inventory, safe_stock,
                           full_penalty, empty_penalty, dist, tau,
                           truck_capacity, cost_per_km, transport_cost_mode='per_trip',
                           service_weight=0.0, solver_time_limit=300):

    prob = LpProblem("Bike_Dispatch_Optimization", LpMinimize)

    I = LpVariable.dicts(
        'I',
        [(i, t) for i in site_ids for t in range(25)],
        lowBound=0,
        cat=LpContinuous
    )

    uB = LpVariable.dicts(
        'uB',
        [(i, t) for i in site_ids for t in hours],
        lowBound=0,
        cat=LpContinuous
    )

    uR = LpVariable.dicts(
        'uR',
        [(i, t) for i in site_ids for t in hours],
        lowBound=0,
        cat=LpContinuous
    )

    q = LpVariable.dicts(
        'q',
        [(i, t) for i in site_ids for t in hours],
        lowBound=0,
        cat=LpContinuous
    )

    x_keys = [(i, j, t) for i in site_ids for j in site_ids if i != j for t in hours]

    x = LpVariable.dicts(
        'x',
        x_keys,
        lowBound=0,
        cat=LpInteger
    )

    m = LpVariable.dicts(
        'm',
        x_keys,
        lowBound=0,
        cat=LpInteger
    )

    if transport_cost_mode == 'per_trip':
        transport_cost = lpSum(cost_per_km * dist[(i, j)] * m[(i, j, t)] for (i, j, t) in x_keys)
    else:

        transport_cost = lpSum(cost_per_km * dist[(i, j)] * x[(i, j, t)] for (i, j, t) in x_keys)

    full_penalty_cost = lpSum(full_penalty[i] * uR[(i, t)] for i in site_ids for t in hours)
    empty_penalty_cost = lpSum(empty_penalty[i] * q[(i, t)] for i in site_ids for t in hours)

    service_loss = lpSum(uB[(i, t)] + uR[(i, t)] for i in site_ids for t in hours)

    prob += transport_cost + full_penalty_cost + empty_penalty_cost + service_weight * service_loss

    for i in site_ids:
        prob += I[(i, 0)] == init_inventory[i], f"init_inventory_{i}"

    for (i, j, t) in x_keys:
        prob += x[(i, j, t)] <= truck_capacity * m[(i, j, t)], f"truck_cap_{i}_{j}_{t}"

    for t in hours:
        for i in site_ids:

            arrival_expr = lpSum(
                x[(j, i, t - tau[(j, i)])]
                for j in site_ids
                if j != i and (t - tau[(j, i)] >= 0)
            )

            out_expr = lpSum(
                x[(i, j, t)]
                for j in site_ids
                if j != i
            )

            prob += out_expr <= I[(i, t)] + arrival_expr, f"out_limit_{i}_{t}"

            prob += uB[(i, t)] >= B[(i, t)] - (I[(i, t)] + arrival_expr - out_expr), f"borrow_fail_lb_{i}_{t}"
            prob += uB[(i, t)] <= B[(i, t)], f"borrow_fail_ub_{i}_{t}"

            prob += (
                I[(i, t + 1)] ==
                I[(i, t)] + arrival_expr - out_expr
                - (B[(i, t)] - uB[(i, t)])
                + (R[(i, t)] - uR[(i, t)])
            ), f"inventory_balance_{i}_{t}"

            prob += (
                uR[(i, t)] >=
                I[(i, t)] + arrival_expr - out_expr
                - (B[(i, t)] - uB[(i, t)])
                + R[(i, t)] - capacity[i]
            ), f"return_fail_lb_{i}_{t}"

            prob += uR[(i, t)] <= R[(i, t)], f"return_fail_ub_{i}_{t}"

            prob += I[(i, t + 1)] <= capacity[i], f"capacity_ub_{i}_{t}"
            prob += I[(i, t + 1)] >= 0, f"capacity_lb_{i}_{t}"

            prob += q[(i, t)] >= safe_stock[i] - I[(i, t + 1)], f"safe_gap_{i}_{t}"

    solver = HiGHS(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    print("求解状态：", LpStatus[prob.status])

    plan_rows = []
    for (i, j, t) in x_keys:
        x_val = value(x[(i, j, t)])
        if x_val is not None and x_val > 1e-6:
            m_val = value(m[(i, j, t)])
            arrive_time = t + tau[(i, j)]
            if transport_cost_mode == 'per_trip':
                tr_cost = cost_per_km * dist[(i, j)] * m_val
            else:
                tr_cost = cost_per_km * dist[(i, j)] * x_val

            plan_rows.append({
                '时间': t,
                '出发站': i,
                '目标站': j,
                '车辆数': x_val,
                '车次': m_val,
                '距离(km)': dist[(i, j)],
                '调度耗时(小时)': tau[(i, j)],
                '到达时间': arrive_time,
                '该次运输成本': tr_cost
            })

    plan_df = pd.DataFrame(plan_rows).sort_values(['时间', '出发站', '目标站']) if len(plan_rows) > 0 else pd.DataFrame(columns=['时间','出发站','目标站','车辆数','车次','距离(km)','调度耗时(小时)','到达时间','该次运输成本'])

    detail_rows = []
    total_demand = 0.0
    borrow_fail_count = 0.0
    return_fail_count = 0.0
    full_occ = 0
    empty_occ = 0

    for t in hours:
        for i in site_ids:
            arrival_val = 0.0
            for j in site_ids:
                if j != i and (t - tau[(j, i)] >= 0):
                    arrival_val += value(x[(j, i, t - tau[(j, i)])])

            out_val = 0.0
            for j in site_ids:
                if j != i:
                    out_val += value(x[(i, j, t)])

            start_inv = value(I[(i, t)])
            end_inv = value(I[(i, t + 1)])
            ub = value(uB[(i, t)])
            ur = value(uR[(i, t)])
            gap = value(q[(i, t)])

            need_b = B[(i, t)]
            need_r = R[(i, t)]
            total_demand += need_b + need_r
            borrow_fail_count += ub
            return_fail_count += ur

            if ur > 1e-6:
                full_occ += 1
            if gap > 1e-6:
                empty_occ += 1

            success_b = need_b - ub
            success_r = need_r - ur

            detail_rows.append({
                'site_id': i,
                'hour': t,
                'start_inventory': start_inv,
                'arrival_dispatch': arrival_val,
                'out_dispatch': out_val,
                'pred_borrow': need_b,
                'pred_return': need_r,
                'success_borrow': success_b,
                'success_return': success_r,
                'borrow_fail': ub,
                'return_fail': ur,
                'end_inventory': end_inv,
                'low_gap': gap
            })

    detail_df = pd.DataFrame(detail_rows)

    transport_cost_val = value(transport_cost)
    full_penalty_val = value(full_penalty_cost)
    empty_penalty_val = value(empty_penalty_cost)
    total_cost_val = value(prob.objective)

    success_total = total_demand - borrow_fail_count - return_fail_count
    demand_rate = 100 * success_total / total_demand if total_demand > 0 else 0

    summary = {
        '方案': '有调度_MILP',
        '运输成本': transport_cost_val,
        '满桩惩罚成本': full_penalty_val,
        '空桩惩罚成本': empty_penalty_val,
        '日总费用': total_cost_val,
        '满桩发生次数': full_occ,
        '空桩发生次数': empty_occ,
        '用户无法借车次数': borrow_fail_count,
        '用户无法还车次数': return_fail_count,
        '用户需求率(%)': demand_rate
    }

    return summary, detail_df, plan_df, prob

summary_no, detail_no = simulate_no_dispatch(
    site_ids=site_ids,
    hours=HOURS,
    B=B,
    R=R,
    capacity=capacity,
    init_inventory=init_inventory,
    safe_stock=safe_stock,
    full_penalty=full_penalty,
    empty_penalty=empty_penalty
)

summary_opt, detail_opt, plan_opt, prob = optimize_dispatch_milp(
    site_ids=site_ids,
    hours=HOURS,
    B=B,
    R=R,
    capacity=capacity,
    init_inventory=init_inventory,
    safe_stock=safe_stock,
    full_penalty=full_penalty,
    empty_penalty=empty_penalty,
    dist=dist,
    tau=tau,
    truck_capacity=TRUCK_CAPACITY,
    cost_per_km=COST_PER_KM,
    transport_cost_mode=TRANSPORT_COST_MODE,
    service_weight=SERVICE_WEIGHT,
    solver_time_limit=SOLVER_TIME_LIMIT
)

compare_df = pd.DataFrame([summary_no, summary_opt])

print("\n=== 对比结果 ===")
print(compare_df)

compare_df.to_csv('问题3_无调度与有调度对比结果.csv', index=False, encoding='utf-8-sig')
detail_no.to_csv('问题3_无调度逐时逐站点结果.csv', index=False, encoding='utf-8-sig')
detail_opt.to_csv('问题3_有调度逐时逐站点结果.csv', index=False, encoding='utf-8-sig')
plan_opt.to_csv('问题3_最优调度计划.csv', index=False, encoding='utf-8-sig')

print("\n文件已导出：")
print("1. 问题3_无调度与有调度对比结果.csv")
print("2. 问题3_无调度逐时逐站点结果.csv")
print("3. 问题3_有调度逐时逐站点结果.csv")
print("4. 问题3_最优调度计划.csv")
