"""
WandB 數據導出工具
從 WandB 抓取所有 project 和 run 的數據，並以獨立的 HTML 格式保存
生成的 HTML 包含完整的數據和圖表，無需登入即可查看
"""
import wandb
from pathlib import Path
import json
import pandas as pd


def convert_to_serializable(obj):
    """
    將 WandB 對象轉換為可序列化的標準 Python 類型
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # 對於其他類型，嘗試轉換為字符串
        return str(obj)


def generate_standalone_html(run, run_name, run_id):
    """
    生成獨立的 HTML 文件，包含完整數據和圖表，無需登入即可查看
    """
    # 獲取 run 的配置、摘要和歷史數據
    config = convert_to_serializable(dict(run.config))
    summary = convert_to_serializable(dict(run.summary))

    # 獲取訓練歷史數據
    # 使用 scan_history() 來獲取完整的歷史數據
    try:
        history_list = []
        for row in run.scan_history():
            # 轉換為可序列化的格式
            serializable_row = convert_to_serializable(dict(row))
            history_list.append(serializable_row)
        history_data = history_list
    except Exception:
        # 如果 scan_history 失敗，嘗試使用 history()
        try:
            history_df = run.history(samples=10000)  # 增加樣本數
            history_data = convert_to_serializable(history_df.to_dict('records'))
        except:
            history_data = []

    # 生成 HTML 模板
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{run_name} ({run_id})</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #ff6b6b;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #4ecdc4;
            padding-bottom: 8px;
        }}
        .info-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .info-item {{
            margin: 8px 0;
            padding: 5px;
        }}
        .info-label {{
            font-weight: bold;
            color: #666;
            display: inline-block;
            min-width: 150px;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 10px;
            background-color: #fafafa;
            border-radius: 5px;
            min-height: 400px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
            clear: both;
        }}
        #charts {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        @media (max-width: 1200px) {{
            #charts {{
                grid-template-columns: 1fr;
            }}
        }}
        .summary-card {{
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-card .key {{
            font-weight: bold;
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 20px;
            color: #333;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{run_name}</h1>
        <div class="info-section">
            <div class="info-item">
                <span class="info-label">Run ID:</span>
                <span>{run_id}</span>
            </div>
            <div class="info-item">
                <span class="info-label">State:</span>
                <span>{run.state}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Created:</span>
                <span>{run.created_at}</span>
            </div>
        </div>

        <h2>Summary Metrics</h2>
        <div class="summary-grid" id="summary-grid"></div>

        <h2>Training History</h2>
        <div id="charts"></div>

        <h2>Configuration</h2>
        <pre>{json.dumps(config, indent=2, ensure_ascii=False)}</pre>
    </div>

    <script>
        // 數據
        const historyData = {json.dumps(history_data, ensure_ascii=False)};
        const summary = {json.dumps(summary, ensure_ascii=False)};

        // 渲染 Summary Cards
        const summaryGrid = document.getElementById('summary-grid');
        Object.entries(summary).forEach(([key, value]) => {{
            if (typeof value === 'number' || typeof value === 'string') {{
                const card = document.createElement('div');
                card.className = 'summary-card';
                card.innerHTML = `
                    <div class="key">${{key}}</div>
                    <div class="value">${{typeof value === 'number' ? value.toFixed(4) : value}}</div>
                `;
                summaryGrid.appendChild(card);
            }}
        }});

        // 獲取所有數值列
        const columns = historyData.length > 0 ? Object.keys(historyData[0]) : [];
        const numericColumns = columns.filter(col => {{
            return historyData.some(row => typeof row[col] === 'number');
        }}).filter(col => col !== '_step' && col !== '_runtime' && col !== '_timestamp');

        // 按指標類型分組
        const metricGroups = {{}};
        numericColumns.forEach(col => {{
            // 提取基礎指標名稱（去掉 train_, eval_, test_ 等前綴）
            let baseMetric = col;
            let prefix = '';

            if (col.startsWith('train/')) {{
                prefix = 'train';
                baseMetric = col.substring(6);
            }} else if (col.startsWith('eval/')) {{
                prefix = 'eval';
                baseMetric = col.substring(5);
            }} else if (col.startsWith('test/')) {{
                prefix = 'test';
                baseMetric = col.substring(5);
            }} else if (col.startsWith('train_')) {{
                prefix = 'train';
                baseMetric = col.substring(6);
            }} else if (col.startsWith('eval_')) {{
                prefix = 'eval';
                baseMetric = col.substring(5);
            }} else if (col.startsWith('test_')) {{
                prefix = 'test';
                baseMetric = col.substring(5);
            }}

            if (!metricGroups[baseMetric]) {{
                metricGroups[baseMetric] = [];
            }}
            metricGroups[baseMetric].push({{name: col, prefix: prefix}});
        }});

        // 為每個指標組創建圖表
        const chartsDiv = document.getElementById('charts');
        Object.entries(metricGroups).forEach(([baseMetric, metrics]) => {{
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            chartDiv.id = 'chart-' + baseMetric.replace(/[^a-zA-Z0-9]/g, '_');
            chartsDiv.appendChild(chartDiv);

            const traces = [];
            metrics.forEach(metric => {{
                // 優先使用 epoch 作為 X 軸，如果沒有則使用 _step
                const x = historyData.map((row, idx) => {{
                    if (row['train/epoch'] !== undefined && row['train/epoch'] !== null) return row['train/epoch'];
                    if (row['epoch'] !== undefined && row['epoch'] !== null) return row['epoch'];
                    if (row['_step'] !== undefined) return row['_step'];
                    return idx;
                }});
                const y = historyData.map(row => row[metric.name]);

                const trace = {{
                    x: x,
                    y: y,
                    mode: 'lines+markers',
                    name: metric.name,
                    line: {{
                        width: 2
                    }},
                    marker: {{
                        size: 4
                    }},
                    connectgaps: true
                }};
                traces.push(trace);
            }});

            // 決定 X 軸標籤
            const hasEpoch = historyData.some(row => row['train/epoch'] !== undefined || row['epoch'] !== undefined);
            const layout = {{
                title: baseMetric,
                xaxis: {{
                    title: hasEpoch ? 'Epoch' : 'Step',
                    gridcolor: '#e0e0e0'
                }},
                yaxis: {{
                    title: 'Value',
                    gridcolor: '#e0e0e0'
                }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#fafafa',
                margin: {{
                    l: 60,
                    r: 30,
                    t: 50,
                    b: 50
                }},
                showlegend: true,
                legend: {{
                    x: 1.05,
                    y: 1
                }}
            }};

            Plotly.newPlot('chart-' + baseMetric.replace(/[^a-zA-Z0-9]/g, '_'), traces, layout, {{responsive: true}});
        }});

        // 如果沒有分組的指標，單獨顯示
        if (Object.keys(metricGroups).length === 0 && numericColumns.length > 0) {{
            const hasEpoch = historyData.some(row => row['train/epoch'] !== undefined || row['epoch'] !== undefined);

            numericColumns.forEach(col => {{
                // 優先使用 epoch 作為 X 軸，如果沒有則使用 _step
                const x = historyData.map((row, idx) => {{
                    if (row['train/epoch'] !== undefined && row['train/epoch'] !== null) return row['train/epoch'];
                    if (row['epoch'] !== undefined && row['epoch'] !== null) return row['epoch'];
                    if (row['_step'] !== undefined) return row['_step'];
                    return idx;
                }});
                const y = historyData.map(row => row[col]);

                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart-container';
                chartDiv.id = 'chart-' + col;
                chartsDiv.appendChild(chartDiv);

                const trace = {{
                    x: x,
                    y: y,
                    mode: 'lines+markers',
                    name: col,
                    line: {{
                        width: 2
                    }},
                    marker: {{
                        size: 4
                    }},
                    connectgaps: true
                }};

                const layout = {{
                    title: col,
                    xaxis: {{
                        title: hasEpoch ? 'Epoch' : 'Step',
                        gridcolor: '#e0e0e0'
                    }},
                    yaxis: {{
                        title: 'Value',
                        gridcolor: '#e0e0e0'
                    }},
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#fafafa',
                    margin: {{
                        l: 60,
                        r: 30,
                        t: 50,
                        b: 50
                    }}
                }};

                Plotly.newPlot('chart-' + col, [trace], layout, {{responsive: true}});
            }});
        }}
    </script>
</body>
</html>
"""
    return html_template


def export_wandb_data_to_html(entity_name=None, output_dir="Training/wandb_latest"):
    """
    從 WandB 抓取所有 project 和 run 的數據，並以獨立 HTML 格式保存
    生成的 HTML 包含完整數據和圖表，無需登入即可查看

    Args:
        entity_name: WandB entity 名稱 (用戶名或組織名)。如果不提供，會使用默認 entity
        output_dir: 輸出目錄路徑
    """
    # 初始化 WandB API
    api = wandb.Api()

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"開始抓取 WandB 數據...")
    print(f"輸出目錄: {output_path.absolute()}")

    # 獲取所有 projects
    if entity_name:
        projects = api.projects(entity=entity_name)
        print(f"\n正在抓取 entity '{entity_name}' 的所有 projects...")
    else:
        # 如果沒有指定 entity，嘗試獲取默認 entity 的 projects
        entity_name = api.default_entity
        projects = api.projects(entity=entity_name)
        print(f"\n使用默認 entity '{entity_name}'")

    project_count = 0
    total_run_count = 0

    # 遍歷所有 projects
    for project in projects:
        project_count += 1
        project_name = project.name
        print(f"\n{'='*60}")
        print(f"Project {project_count}: {project_name}")
        print(f"{'='*60}")

        # 為每個 project 創建子目錄
        project_dir = output_path / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # 獲取該 project 下的所有 runs
        runs_path = f"{entity_name}/{project_name}"
        runs = api.runs(runs_path)

        run_count = 0
        # 遍歷所有 runs
        for run in runs:
            run_count += 1
            total_run_count += 1
            run_id = run.id
            run_name = run.name

            print(f"  [{run_count}] Run: {run_name} (ID: {run_id})")

            try:
                # 生成獨立的 HTML（包含完整數據和圖表）
                html_content = generate_standalone_html(run, run_name, run_id)

                # 構建安全的文件名 (使用 run_name 和 run_id 以確保唯一性)
                safe_run_name = "".join(
                    c for c in run_name
                    if c.isalnum() or c in (' ', '-', '_')
                ).strip()
                safe_run_name = safe_run_name.replace(' ', '_')
                filename = f"{safe_run_name}_{run_id}.html"
                file_path = project_dir / filename

                # 保存 HTML 文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                print(f"      [OK] 已保存: {file_path.relative_to(output_path)}")

            except Exception as e:
                print(f"      [ERROR] 錯誤: {str(e)}")
                continue

        print(f"\n  Project '{project_name}' 完成: {run_count} 個 runs")

    # 輸出總結
    print(f"\n{'='*60}")
    print(f"抓取完成！")
    print(f"總共處理: {project_count} 個 projects, {total_run_count} 個 runs")
    print(f"所有數據已保存至: {output_path.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 使用方式 1: 使用默認 entity
    # export_wandb_data_to_html()

    # 使用方式 2: 指定 entity 名稱
    # export_wandb_data_to_html(entity_name="your-entity-name")

    # 使用方式 3: 指定 entity 和輸出目錄
    export_wandb_data_to_html(
        entity_name="chsyu-national-chengchi-university",
        output_dir="Training\wandb_latest"
    )
