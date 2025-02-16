import pandas as pd
import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import json
import plotly.express as px
# lsof -n -i :8050 | grep LISTEN
# kill -9 $(lsof -t -i :8050)
try:
    from .mlops_data_interface import SqliteDataInterface
except:
    from mlops_data_interface import SqliteDataInterface

import subprocess
import time

data_interface = SqliteDataInterface()

def kill_process_on_port(port, timeout=10):
    # Find the PID of the process listening on the specified port
    find_pid_cmd = f"lsof -t -i :{port}"
    
    try:
        # Run the command to get the PID
        pid = subprocess.check_output(find_pid_cmd, shell=True).decode().strip()
        
        if pid:
            # If a PID is found, kill the process
            kill_cmd = f"kill -9 {pid}"
            subprocess.run(kill_cmd, shell=True)
            print(f"Process {pid} on port {port} has been terminated.")
            
            # Wait for the process to terminate
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if the process is still running
                try:
                    subprocess.check_output(f"ps -p {pid}", shell=True)
                    time.sleep(0.5)  # Wait a bit before checking again
                except subprocess.CalledProcessError:
                    print(f"Process {pid} has been successfully terminated.")
                    return
            print(f"Process {pid} did not terminate within {timeout} seconds.")
        else:
            print(f"No process found on port {port}.")
    
    except subprocess.CalledProcessError:
        print(f"No process is listening on port {port}.")




df = data_interface.get_data_for_dashboard()
PAGE_SIZE = 12


app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = html.Div(
    className="container",
    style={'backgroundColor': '#0066CC'},  # Light orange color for the background
    children = [
        html.Div(
            className="jumbotron text-center",  # Bootstrap Jumbotron for header
            children=[
                html.H1("Models Registry", className="display-4"),
                html.P("An interactive dashboard for model performance", className="lead"),
            ],
            style={'marginBottom': '40px',
                   'color': 'white'}
        ),
        html.Div(
            className="row mb-4",  # Bootstrap row and margin classes
            children=[
                html.Div(
                    className="col",  # Bootstrap column class
                    children=[
                        dcc.Input(
                            value="", id="filter-input1", placeholder="Include",
                            debounce=True, className="form-control"
                        ),
                    ],
                ),
                html.Div(
                    className="col",  # Bootstrap column class
                    children=[
                        dcc.Input(
                            value="", id="filter-input2", placeholder="Exclude",
                            debounce=True, className="form-control"
                        ),
                    ],
                ),
            ]
        ),
        html.Div(
            dash_table.DataTable(
                id="datatable-paging",
                columns=[{"name": i, "id": i} for i in df.columns],  # sorted(df.columns)
                page_current=0,
                page_size=PAGE_SIZE,
                page_action="custom",
                sort_action="custom",
                sort_mode="single",
                sort_by=[],
                row_selectable='single',  # Enable single row selection
                selected_rows=[],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }],
                style_cell_conditional=[
                    {'if': {'column_id': 'experiment_timestamp'}, 'width': '20%'},
                    {'if': {'column_id': 'experiment_name'}, 'width': '25%'},
                    {'if': {'column_id': 'tags'}, 'width': '30%'},
                    {'if': {'column_id': 'metrics'}, 'width': '15%'},
                    {'if': {'column_id': 'ram_footprint_gb'}, 'width': '5%'},
                    {'if': {'column_id': 'classes_nr'}, 'width': '5%'},
                ],
            ),
            className="table table-striped table-bordered",
            style={
                'margin': '0 auto',  # Center the table
                # 'margin': '20px auto',  # Add margin around the table
                'maxWidth': '100%',   # Set the maximum width of the table
                # 'padding': '10px',   # Add padding around the table
                'textAlign': 'center',  # Center-align the content
                'overflowX': 'auto',  # Add horizontal scroll if content exceeds the width
            }
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="row-graph1"), style={'width': '70%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="row-graph3"), style={'width': '30%', 'display': 'inline-block'}),
            ],
            style={'display': 'flex', 'width': '100%'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="row-graph2"), style={'width': '70%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id="row-graph5"), style={'width': '30%', 'display': 'inline-block'}),
            ],
            style={'display': 'flex', 'width': '100%'}
        ),
        dcc.Graph(id="row-graph4"),
        dcc.Graph(id="row-graph6"),
        html.Div([
            dcc.ConfirmDialog(
                id='confirm-danger',
                message='Danger danger! Are you sure you want to continue?',
            ),

            html.Button('Delete', id='delete-button', n_clicks=0),
            html.Div(id='output-danger')
        ])
    ]
)


@app.callback(Output('confirm-danger', 'displayed'),
              Input('delete-button', 'n_clicks'))
def display_confirm(n_clicks):
    if n_clicks:
        return True
    return False

@app.callback(Output('output-danger', 'children'),
              [Input('confirm-danger', 'submit_n_clicks'),Input("datatable-paging", "selected_rows"),Input("datatable-paging", "data")])
def update_output(submit_n_clicks,selected_rows, table_data):
    if submit_n_clicks:
        if selected_rows:
            # Get the data of the selected row
            selected_row = selected_rows[0]
            try:
                row_data = table_data[selected_row]
            except IndexError:
                return dash.no_update
            timestamp = row_data['experiment_timestamp'].replace('T',' ')
            print(f"Deleting row with timestamp: {timestamp}")
            data_interface.delete_experiment(timestamp)
    return 'It wasnt easy but we did it'
    
@app.callback(
    Output("datatable-paging", "data"),
    [
        Input("datatable-paging", "page_current"),
        Input("datatable-paging", "page_size"),
        Input("datatable-paging", "sort_by"),
        Input("filter-input1", "value"),
        Input("filter-input2", "value"),
    ],
)
def update_table(page_current, page_size, sort_by, include, exclude):
    # Filter
    dff = data_interface.get_data_for_dashboard(filter_include=include if include else None,
                                                filter_exclude=exclude if exclude else None)

    # Sort if necessary
    if len(sort_by):
        dff = dff.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )

    return dff.iloc[page_current * page_size : (page_current + 1) * page_size].to_dict("records")

@app.callback(
    [Output("row-graph1", "figure"),
     Output("row-graph2", "figure"),
     Output("row-graph3", "figure"),
     Output("row-graph4", "figure"),
     Output("row-graph5", "figure"),
     Output("row-graph6", "figure")],
    [Input("datatable-paging", "selected_rows"),
     Input("datatable-paging", "data")]
)
def update_graph(selected_rows, table_data):
    if not selected_rows:
        return dash.no_update
    
    # Get the data of the selected row
    selected_row = selected_rows[0]
    try:
        row_data = table_data[selected_row]
    except IndexError:
        return dash.no_update
    timestamp = row_data['experiment_timestamp'].replace('T',' ')
    df = data_interface.get_data_for_dashboard(timestamp)
    row_data = df.iloc[0]
    if row_data['feature_importance']:
        fi = pd.read_json(row_data['feature_importance'])
        fig4 = px.bar(x=fi['Feature'], y=fi['Importance'])
        fig4.update_layout(title=f"Feature Importance")
    else:
        fig4 = px.bar(x=[0], y=[0])
    cr = pd.read_json(row_data['classification_report'])
    exclude = ['accuracy','macro avg','weighted avg']
    cr = cr[~cr.index.isin(exclude)]
    print(len(cr))
    # Example plot: plotting metrics vs ram_footprint_gb for the selected row
    fig1 = px.bar(x=cr.index, y=cr['f1-score'],labels={'x': 'Class', 'y': 'f1-score'})
    fig1.update_layout(title=f"Experiment: {row_data['experiment_name']}")
    # import pdb;pdb.set_trace()
    fig2 = px.scatter(x=cr['f1-score'], y=cr['support'],labels={'x': 'f1-score', 'y': 'samples'},hover_data=[cr.index],title=f"F1-score vs Nr of samples")
    # fig2.update_traces(title=f"F1-score vs Nr of samples")
    fig2.update_traces(marker=dict(size=12))
    # fig2.update_layout(title=f"Experiment: {row_data['experiment_name']}")
    cr['f1-score_over_.7'] = cr['f1-score'] > 0.7
    fig5 = px.pie(cr, names='f1-score_over_.7', title='f1-score > 0.7')
    fig3 = px.pie(cr, values='support', names=cr.index, title='Samples per class')
    try:
        try:
            a = json.loads(row_data['target_names'])
        except:
            a = cr.index.to_list()
        b = np.array(json.loads(row_data['confusion_matrix']))
        # let's normalize the confusion matrix
        b = b*100/b.sum(axis=1)[:,None]

        fig6 = px.imshow(b, labels=dict(x="Predicted", y="True", color="Percents"), x=a, y=a)
        fig6.update_layout(title=f"Confusion matrix")
    except Exception as e:
        print(e)    
        fig6 = px.imshow(np.array([[0]]), labels=dict(x="Predicted", y="True", color="Percent"), x=[0], y=[0])

    return fig1, fig2, fig3,fig4,fig5,fig6

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except:
        print('Port in use, killing process')
        kill_process_on_port(8050)
        time.sleep(2)
        app.run_server(debug=False)
