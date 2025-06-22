import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

def parse_supervised_metrics(metrics_csv_filename):
    metrics = None
    with open(metrics_csv_filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            if not metrics:
                metrics = { m:[] for m in row }
            else:
                for i, (k,v) in enumerate(metrics.items()):
                    if row[i] != "":
                        if i >= 2:
                            # Add epoch, step, metric_info
                            v.append( (int(row[0]), int(row[1]), float(row[i])) )
                        else:
                            # Add epoch or step info
                            v.append( int(row[i]) )
    return metrics

def parse_eval_metrics(metrics_csv_filename):
    metrics = None
    with open(metrics_csv_filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            if not metrics:
                metrics = { m:[] for m in row }
            else:
                for i, (k, v) in enumerate(metrics.items()):
                    if row[0] != "" and row[i] != "":
                        v.append((int(row[0]), int(row[2]), float(row[i])))

    return metrics

def plot_results_plotly(sorted_list, stats):
    color_array = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    N = len(sorted_list)
    
    # Create subplot grid with shared y-axis
    fig = make_subplots(rows=1, cols=N, shared_yaxes=True, subplot_titles=sorted_list)
    
    for i, basename in enumerate(sorted_list):
        values = stats[basename]
        for j, (version, metrics) in enumerate(values.items()):
            val_stats = metrics["val_accuracy"]
            epochs, steps, val_accs = zip(*val_stats)
            
            # Line plot for validation accuracy
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=val_accs,
                    mode='lines',
                    name=f"Val: {version}",
                    line=dict(color=color_array[j % len(color_array)]),
                    legendgroup=f"{basename}_{version}",
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
            
            # Horizontal line for test accuracy
            if "test_accuracy" in metrics:
                test_y = metrics["test_accuracy"][-1][2]
                fig.add_trace(
                    go.Scatter(
                        x=[min(steps), max(steps)],
                        y=[test_y, test_y],
                        mode='lines',
                        name=f"Test: {version}",
                        line=dict(
                            color=color_array[j % len(color_array)],
                            dash='dash'
                        ),
                        legendgroup=f"{basename}_{version}",
                        showlegend=(i == 0)
                    ),
                    row=1, col=i+1
                )

    # Update layout
    fig.update_layout(
        height=400,
        width=N * 400,
        font=dict(size=10),
        annotations=[
            dict(
                x=ann['x'],
                y=ann['y'],
                xref=ann['xref'],
                yref=ann['yref'],
                text=ann['text'],
                showarrow=False,
                font=dict(size=12)
            ) for ann in fig['layout']['annotations']
        ]
    )

    # Y-axis label (only once for shared axis)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    
    # X-axis label for all
    for i in range(N):
        fig.update_xaxes(title_text="Steps", row=1, col=i+1)
    
    fig.show()