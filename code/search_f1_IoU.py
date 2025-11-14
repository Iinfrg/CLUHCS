def parse_file(filename, top_n=5):
    results = []

    with open(filename, 'r') as f:
        lines = f.readlines()

        # 每次读取3行为一组
        for i in range(0, len(lines), 3):
            # 解析指标行
            metrics_line = lines[i].strip()
            # 解析参数行
            params_line = lines[i + 1].strip() if i + 1 < len(lines) else ''

            # 跳过格式不完整的组
            if not metrics_line.startswith('precision=') or not params_line.startswith('w='):
                continue

            # 解析指标
            metrics = {}
            for item in metrics_line.split(', '):
                key, value = item.split('=')
                metrics[key.strip()] = float(value)

            # 解析参数
            params = {}
            for item in params_line.split(', '):
                key, value = item.split('=')
                params[key.strip()] = float(value) if '.' in value else int(value)

            # 保存完整记录
            results.append({
                'f1': metrics['f1_score'],
                'metrics': metrics,
                'params': params
            })

    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:top_n]

    print(f"Top {top_n} f1_score记录：")
    for idx, entry in enumerate(sorted_results, 1):
        print(f"\n第{idx}名：")
        print(f"f1_score={entry['f1']:.4f}")
        print("对应指标：")
        print(', '.join([f"{k}={v:.4f}" for k, v in entry['metrics'].items()]))
        print("对应参数：")
        print(', '.join([f"{k}={v}" for k, v in entry['params'].items()]))
        print("-" * 50)
    return sorted_results


def search_top_f1_by_layer(filename, target_layer, top_n=5):
    candidates = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            metrics_line = lines[i].strip()
            params_line = lines[i + 1].strip() if i + 1 < len(lines) else ''

            if not metrics_line.startswith('precision=') or not params_line.startswith('w='):
                continue

            # 解析指标
            metrics = {}
            for item in metrics_line.split(', '):
                key, value = item.split('=')
                metrics[key.strip()] = float(value)

            # 解析参数
            params = {}
            for item in params_line.split(', '):
                key, value = item.split('=')
                params[key.strip()] = float(value) if '.' in value else int(value)

            # 筛选目标层数
            if params.get('t_n_layers') == target_layer:
                entry = {
                    'f1': metrics['f1_score'],
                    'metrics': metrics,
                    'params': params
                }
                candidates.append(entry)

    sorted_candidates = sorted(candidates, key=lambda x: -x['f1'])[:top_n]
    print(f"\n【t_n_layers={target_layer} 的前{top_n}名F1记录】")
    if sorted_candidates:
        for idx, entry in enumerate(sorted_candidates, 1):
            print(f"第{idx}名: f1_score={entry['f1']:.4f}")
            print("指标:", ', '.join([f"{k}={v:.4f}" for k, v in entry['metrics'].items()]))
            print("参数:", ', '.join([f"{k}={v}" for k, v in entry['params'].items()]))
            print("-" * 50)
    else:
        print(f"未找到t_n_layers={target_layer}的记录")
    return sorted_candidates

if __name__ == "__main__":
    import sys
    argv = sys.argv
    dataset = argv[1]  # 拿到数据集名称
    top_n = 6  # 设置需要输出的前n名
    averesult = f"./embeds/{dataset}" + "/ave_result.txt"
    target_layer = 4  # 指定要查询的t_n_layers值
    search_top_f1_by_layer(averesult, target_layer, top_n=top_n)

