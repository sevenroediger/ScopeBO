import base64
from collections import OrderedDict
from datetime import datetime
import hashlib
import io
from io import BytesIO
import os
import threading
from threading import Thread
import socket
import time
import traceback
import webbrowser

import dash
from dash import callback, dcc, html, Input, Output, State, no_update, ALL, ctx, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw

from Code.predictor import ScopeBO
from Code.featurization import calculate_morfeus_descriptors_web, build_common_core_preview
from Code.space_creator import create_search_space_web


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

app.title = "ScopeBO"

# In-memory cache for rendered SMILES PNG data URIs to keep hover tooltips snappy.
_SMILES_IMG_CACHE = OrderedDict()
_SMILES_IMG_CACHE_MAX = 10000

# Background featurization job state shared by callbacks.
_FEATURIZATION_JOBS = {}
_FEATURIZATION_JOBS_LOCK = threading.Lock()

# Color scheme
navbar_color = "#1561C2"
header_color = navbar_color
color_good = "#188F9D"
color_bad = "#CE4C6F"
color_warning = "#FFA618"
background_color = "#E5E5E5"
border_color = "#AFAFAF"
back_button_color = "#797979"
success_color = "#3AB8C6"
danger_color = "#CEA8B2"
info_color = "#515798"

# Style templates
BUTTON_STYLE_GOOD = {
    "backgroundColor": "#A3D2D7",
    "borderColor": color_good,
    "borderWidth": "1px",
    "color": color_good,
    "fontSize": "16px",
    "fontWeight": "bold",
    "padding": "3px 10px",
}

BUTTON_STYLE_BAD = {
    "backgroundColor": "#CEA8B2",
    "borderColor": color_bad,
    "borderWidth": "1px",
    "color": color_bad,
    "fontSize": "16px",
    "fontWeight": "bold",
    "padding": "3px 10px",
}

UPLOAD_STYLE = {
    "width": "100%",
    "height": "40px",
    "lineHeight": "40px",
    "border": "1px solid ",
    "borderRadius": "4px",
    "borderColor": border_color,
    "borderWidth": "1px",
    "backgroundColor": background_color,
    "textAlign": "center",
    "paddingLeft": "12px",
    "cursor": "pointer",
    "margin": "0 auto",
    "overflow": "hidden",
    "whiteSpace": "nowrap",
    "textOverflow": "ellipsis",
}

INNER_CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "0.25rem",
    "boxShadow": "0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)",
}

OUTER_CARD_STYLE = {
    "backgroundColor": background_color,
    "borderColor": border_color,
    "borderRadius": "0.25rem",
    "boxShadow": "0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)",
    "padding": "0.5% 0.5%",
    "marginBottom": "1rem",
}

HEADER_STYLE = {
    "backgroundColor": header_color,
    "color": "white",
    "fontWeight": "600",
}

MAIN_TAB_STYLE = {
    "backgroundColor": background_color,
    "color": navbar_color,
    "fontSize": "16px",
    "fontWeight": "bold",
    "borderColor": background_color,
}

MAIN_TAB_ACTIVE_STYLE = {
    "backgroundColor": background_color,
    "color": border_color,
    "fontSize": "16px",
    "fontWeight": "bold",
    "borderColor": border_color,
    "borderBottom": f"3px solid white",

}

MAIN_TAB_LABEL_STYLE = {
    "color": back_button_color,
}

MAIN_TAB_ACTIVE_LABEL_STYLE = {
    "color": navbar_color,
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "0.375rem 0.75rem",
    "border": f"1px solid",
    "borderRadius": "0.25rem",
    "backgroundColor": background_color,
    "borderColor": border_color,
}

DROPDOWN_STYLE = {
    "width": "100%",
    "textAlign": "left",
    "padding": "0.375rem 2rem",
    "border": f"1px solid {border_color}",
    "borderRadius": "0.25rem",
    "backgroundColor": background_color,
    "borderColor": border_color,
}

PRIM_BUTTON_STYLE = {
    "backgroundColor": "primary",
    "borderColor": "primary",
    "borderWidth": "3px",
    "color": "white",
    "fontSize": "16px",
    "fontWeight": "bold",
    "padding": "3px 10px",
}

SUCCESS_ALERT_STYLE = {
    "backgroundColor": success_color,
    "borderColor": success_color,
    "color": "white",
    "fontSize": "16px",
    "padding": "3px 10px",
}

DANGER_ALERT_STYLE = {
    "backgroundColor": color_bad,
    "borderColor": color_bad,
    "color": "white",
    "fontSize": "16px",
    "fontWeight": "bold",
    "padding": "3px 10px",
}

WARNING_ALERT_STYLE = {
    "backgroundColor": color_warning,
    "borderColor": color_warning,
    "color": "white",
    "fontSize": "16px",
    "fontWeight": "bold",
    "padding": "3px 10px",
}

INFO_ALERT_STYLE = {
    "backgroundColor": info_color,
    "borderColor": info_color,
    "color": "white",
    "fontSize": "16px",
    "padding": "3px 10px",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": "#f5f5f5",
    "fontWeight": "bold",
    "textAlign": "center",
    "padding": "10px",
    "borderBottom": "2px solid #ddd",
    "position": "sticky",
    "top": "0",
    "zIndex": "1",
}

TABLE_CELL_STYLE = {
    "padding": "8px 12px",
    "borderBottom": "1px solid #eee",
    "verticalAlign": "middle",
}

MODAL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "20px",
    "height": "20px",
    "borderRadius": "50%",
    "backgroundColor": "#0d6efd",
    "borderColor": "#ffffff",
    "color": "white",
    "fontWeight": "bold",
    "fontSize": "12px",
    "cursor": "help",
}


app.layout = dbc.Container([
    # Storage components
    dcc.Store(id='store-search-space'),  # stores the working search space file
    dcc.Store(id='store-objectives', data = [{"name":"Objective 1", "mode": "max"}]),  # stores the selected objectives for optimization
    dcc.Store(id='store-updated', data=True),  # stores if the search space has been updated since the last ScopeBO run
    dcc.Store(id="store-scope-grid"),  # stores the data for the scope grid visualization (scope table)
    dcc.Store(id='store-init', data=True),  # stores if the scope is currently fully unoccupied
    dcc.Store(id='store-pred-results'),  # stores latest prediction dataframe for download
    dcc.Store(id='store-featurization-data', data=[]),  # stores feature datasets
    dcc.Store(id='store-featurization-metadata', data=[]),  # stores table-updated metadata for each dataset in the featurization data store
    dcc.Store(id='store-smiles-upload-valid', data=False),  # tracks whether featurization upload passed validation
    dcc.Store(id='store-umap-figure'),  # stores the most recently generated UMAP figure so it survives tab switches
    dcc.Store(id='store-umap-searchspace-signature'),  # tracks which search space the current UMAP reflects
    dcc.Store(id='store-shap-beeswarm-figure'),  # stores the latest SHAP beeswarm figure across tab switches
    dcc.Store(id='store-shap-bar-figure'),  # stores the latest SHAP bar figure across tab switches
    dcc.Store(id='store-shap-searchspace-signature'),  # tracks which search space the current SHAP plots reflect
    dcc.Store(id='store-pred-figure'),  # stores the most recently generated prediction figure across tab switches
    dcc.Store(id='store-pred-searchspace-signature'),  # tracks which search space the current prediction plot reflects
    dcc.Store(id='store-featurization-trigger'),  # trigger payload to start long featurization run after preview is shown
    dcc.Interval(id='interval-featurization-progress', interval=1000, n_intervals=0, disabled=True),  # interval component to poll for featurization progress
    dcc.Store(
        id='store-other-rows',
        data=[
            {"row_id": 0, "smiles": ""},
        ],
    ),  # editable rows for reporting non-suggested substrates


############################################################
# APP LAYOUT
############################################################

    # App header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                [
                    html.Img(
                        src=dash.get_asset_url("Icon.png"),
                        height="50px",
                        className="me-2 rounded",
                    ),
                    html.Div(
                        [
                            html.Strong("ScopeBO"),
                            # ": Machine Learning-Guided Substrate Scope Selection",
                        ]
                    ),
                ],
                href="/",
                className="ms-2 text-white d-flex align-items-center",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Source available on ",
                                className="text-white small",
                            ),
                            html.A(
                                "GitHub",
                                href="https://github.com/doyle-lab-ucla/ScopeBO",
                                target="_blank",
                                className="text-white small text-decoration-underline",
                            ),
                        ],
                        className="lh-sm",
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Please cite our work: ",
                                className="text-white small",
                            ),
                            html.A(
                                [
                                    "Roediger, S.; Sigman, M. S.; Doyle, A. G. ",
                                    html.I("ChemRxiv "),
                                    html.B("2025"),
                                    " DOI: 10.26434/chemrxiv-2025-r0sst",
                                ],
                                href="https://chemrxiv.org/doi/10.26434/chemrxiv-2025-r0sst",
                                target="_blank",
                                className="text-white small text-decoration-underline",
                            ),
                        ],
                        className="lh-sm",
                    ),
                ],
                className="ms-auto text-end px-3 d-flex flex-column justify-content-center",
            ),
        ], fluid=True),
        color=navbar_color,
        dark=True,
        className="mb-4 py-2",
    ),


    # Starting page
    html.Div(id='page-upload', style={'display': 'block'}, children=[
        dbc.Card([
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.Span("1. Upload search space"),
                                            dbc.Button(
                                                "?",
                                                id="open-searchspace-help",
                                                size="sm",
                                                style=MODAL_STYLE,
                                            ),
                                        ],
                                        className="d-flex justify-content-between align-items-center",
                                    ),
                                    style=HEADER_STYLE,
                                ),
                                dbc.CardBody(children=[
                                    dcc.Upload(id = 'upload-searchspace', children = html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Search Space File')
                                    ], style=UPLOAD_STYLE),
                                    ),
                                    dcc.Loading(id="loading-search-space",
                                                type="default",
                                                children=html.Div(id='feedback-searchspace-upload', className="mt-2")),
                                    html.Div(id="feedback-obj-inference", className="mt-2"),
                                ]),
                            ],
                            className="h-100", style=INNER_CARD_STYLE
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.Span("2. Define objectives"),
                                            dbc.Button(
                                                "?",
                                                id="open-objectives-help",
                                                size="sm",
                                                style=MODAL_STYLE,
                                            ),
                                        ],
                                        className="d-flex justify-content-between align-items-center",
                                    ),
                                    style=HEADER_STYLE,
                                ),
                                dbc.CardBody(children=[
                                    # Section for objective selection (rendered dynamically)
                                    html.Div(id='input-objectives',
                                            style={
                                                "maxWidth": "100%", 
                                                "margin": "0 auto",
                                                "width": "460px"
                                            },
                                    ),
                                    html.Div(children=[
                                        dbc.Button("+", id="btn-add-obj", 
                                                   style={**BUTTON_STYLE_GOOD, "flex": "1"}, 
                                                   className="me-2"),
                                        dbc.Button("-", id="btn-remove-obj", 
                                                   style={**BUTTON_STYLE_BAD, "flex": "1"})
                                    ],
                                    style={
                                        "maxWidth": "100%", 
                                        "width": "460px",
                                        "gap": "5px",
                                        "margin": "0 auto",
                                        "display": "flex",
                                    },
                                    ),
                                ]),
                            ],
                            className="h-100", style=INNER_CARD_STYLE
                        ),
                        width=6,
                    ),
                ],
                className="g-4 align-items-stretch mb-4"
            ),
            html.Div(        
                dbc.Button(
                "Continue to ScopeBO",
                id="btn-search-to-main",
                style=PRIM_BUTTON_STYLE,
                disabled=True,  # only enable once search space is uploaded
                ),
                style={"width": "fit-content", "margin": "0 auto"}
            ),
            dcc.Loading(
                id="loading-search-to-main",
                type="default",
                children=html.Div(id="feedback-obj-status", className="mt-2"),
            ),
        ],
        style=OUTER_CARD_STYLE
        ),
        html.Br(),
            dbc.Card([
                dbc.Card([
                    dbc.CardHeader(
                        html.Div(
                            [
                                html.Span("Don't have a featurized search space yet?"),
                                dbc.Button(
                                    "?",
                                    id="open-feat-overview-help",
                                    size="sm",
                                    style=MODAL_STYLE,
                                ),
                            ],
                            className="d-flex justify-content-between align-items-center",
                        ),
                        style=HEADER_STYLE,
                    ),
                    dbc.CardBody(children=[
                        html.Div(children=[
                            dbc.Button(
                                "1. Featurize SMILES",
                                id="btn-go-to-features",
                                style=PRIM_BUTTON_STYLE,
                                className="me-2",
                            ),
                            dbc.Button(
                                "2. Combine and preprocess features",
                                id="btn-go-to-preprocess",
                                style=PRIM_BUTTON_STYLE,
                                className="ms-2",
                            )],
                            style={"width": "fit-content", "margin": "0 auto"}
                        ),
                    ]),
                ], style=INNER_CARD_STYLE
                ),
            ],style=OUTER_CARD_STYLE),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Search Space Help")),
                dbc.ModalBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Search space file format", 
                                    label_style={"color": navbar_color},
                                    active_label_style={"fontWeight": "bold", "color": navbar_color},
                                    children=[
                                        html.Br(),
                                        dcc.Markdown("- The first column must contain the SMILES strings of the substrates."),
                                        dcc.Markdown("- The subsequent columns must contain the corresponding numerical feature values for each substrate."),
                                        dcc.Markdown("- The first row must contain the feature names, subsequent rows contain the data for the substrates."),
                                        dcc.Markdown("- If you don't have a featurized search space yet, you can use the section at the bottom of the page to generate one from lists of SMILES."),
                                        dcc.Markdown("- The search space must contain at least 58 substrates for ScopeBO to work."),
                                        dcc.Markdown("- While there is no strict upper limit, we recommend a search space size of ca. 300 to 5,000."),
                                        dcc.Markdown("- See the tables in [this ScopeBO example](https://github.com/doyle-lab-ucla/ScopeBO/blob/main/Examples/ScopeBO_example.ipynb) for reference."),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Substrate selection", 
                                    label_style={"color": navbar_color},
                                    active_label_style={"fontWeight": "bold", "color": navbar_color},
                                    children=[
                                        html.Br(),
                                        dcc.Markdown("- The initial substrate pool should be as unbiased as possible."),
                                        dcc.Markdown("- For instance, it could be all commercially available substrates for the compound class (e.g., extracted from Reaxys)."),
                                        dcc.Markdown("- See the [supporting information, section 2.5.1](https://chemrxiv.org/doi/10.26434/chemrxiv-2025-r0sst) of our paper for instructions on extracting substrate lists from Reaxys."),
                                        dcc.Markdown("- If you are aware of incompatible functional groups, you should remove them from the initial search space and report the pruned FGs in addition to your scope."),
                                        dcc.Markdown("- See [this example](https://github.com/doyle-lab-ucla/ScopeBO/blob/main/Examples/Searchspace_sample_pruning_example.ipynb) for a demonstration of how to carry out this curation step."),
                                    ],
                                ),
                            ]
                        )
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-searchspace-help",
                        color="secondary",
                    )
                ),
            ],
            id="searchspace-help-modal",
            className="help-modal",
            is_open=False,
            dialog_style={"width": "90%","maxWidth": "1200px"},
            centered=True,
            scrollable=True,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Objectives Help")),
                dbc.ModalBody(
                    [
                        dcc.Markdown("- You can define one or more objectives for ScopeBO to optimize."),
                        dcc.Markdown("- Provide the objective name in the left input box."),
                        dcc.Markdown("- For each objective, you can choose whether to maximize or minimize it."),
                        dcc.Markdown("- Add or remove objectives using the '+' and '-' buttons below the input boxes."),
                        dcc.Markdown("- If the objective is already present in the search space, it likely has already been added automatically."),
                        dcc.Markdown("- If your objective is a selectivity (e.g., *ee*), we recommend that you provide the values as ddG values instead of the raw selectivity values."),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-objectives-help",
                        color="secondary",
                    )
                ),
            ],
            id="objectives-help-modal",
            className="help-modal",
            is_open=False,
            dialog_style={"width": "90%","maxWidth": "1200px"},
            centered=True,
            scrollable=True,
        ),
                dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Search space creation help")),
                dbc.ModalBody(
                    [
                        dcc.Markdown("If you do not have a featurized search space yet, you can create one here within ScopeBO."),
                        dcc.Markdown("**Feature generation:**"),
                        dcc.Markdown("- You can generate molecular features from a list of SMILES strings."),
                        dcc.Markdown("- If you don't have such a list yet, please see 'Substrate selection' tab in the help for the 'Upload search space' section."),
                        dcc.Markdown("- For a multi-reactant reaction, calculate features separately for each reactant."),
                        dcc.Markdown("- The generated features will be stored in memory and can be downloaded as a CSV file."),
                        dcc.Markdown("- The features are calculated using [Morfeus](https://github.com/digital-chemistry-laboratory/morfeus) at the [GFN2-xTB](https://doi.org/10.1021/acs.jctc.8b01176) level of theory."),
                        dcc.Markdown("- They are calculated for the entire molecule, as well as for the common core atoms of the molecule (if applicable)."),
                        dcc.Markdown("**Preprocessing:**"),
                        dcc.Markdown("- Prepare existing substrate feature sets for ScopeBO by removing highly correlated features."),
                        dcc.Markdown("- Combine feature sets for different reactants into a single search space (the size will be the combinatorial)."),
                        dcc.Markdown("- The resulting search space can be downloaded as a CSV file and then used for substrate suggestions."),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-feat-overview-help",
                        color="secondary",
                    )
                ),
            ],
            id="feat-overview-help-modal",
            className="help-modal",
            is_open=False,
            dialog_style={"width": "90%","maxWidth": "1200px"},
            centered=True,
            scrollable=True,
        )
    ]),


    # Featurization page
    html.Div(id='page-featurization', style={'display': 'none'}, children=[
        dbc.Card(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.Div(
                                [
                                    html.Span("Generate molecular features from a list of SMILES"),
                                    dbc.Button(
                                        "< Back to starting page",
                                        id="btn-back-to-upload-from-featurization",
                                        color="light",
                                        outline=False,
                                        size="sm",
                                        className="py-0",
                                        style={"whiteSpace": "nowrap"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "gap": "0.75rem",
                                },
                            ),
                            style=HEADER_STYLE,
                        ),
                        dbc.CardBody(
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Upload(
                                                    id="upload-smiles",
                                                    children=html.Div(
                                                        [
                                                            "Drag and Drop or ",
                                                            html.A("Select a list of SMILES"),
                                                        ],
                                                        style=UPLOAD_STYLE,
                                                    ),
                                                ),
                                                dbc.Button(
                                                    "Featurize SMILES",
                                                    id="btn-featurization",
                                                    style=PRIM_BUTTON_STYLE,
                                                    className="me-2",
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    "Interrupt featurization",
                                                    id="btn-interrupt-featurization",
                                                    style={"display": "none"},
                                                    outline=False,
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "justifyContent": "center",
                                                "alignItems": "center",
                                                "gap": "12px",
                                            },
                                        ),
                                        html.Div(
                                            id="feedback-upload-smiles",
                                            className="mt-2",
                                            style={"textAlign": "center"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="featurization-running-note",
                                    className="mb-4",
                                ),
                                dcc.Loading(
                                    id="loading-featurization",
                                    type="default",
                                    children=html.Div(
                                        id="feedback-featurization-run",
                                        className="mt-2",
                                    ),
                                ),
                                dbc.Progress(
                                    id="progress-featurization",
                                    value=0,
                                    label="0%",
                                    striped=True,
                                    animated=False,
                                    style={
                                        "height": "22px",
                                        "maxWidth": "540px",
                                        "margin": "0.5rem auto",
                                        "display": "none",
                                    },
                                    className="mt-2",
                                ),
                                html.Div(
                                    id="progress-featurization-text",
                                    className="text-center text-muted small",
                                ),
                                html.Div(
                                    id="common-core-preview",
                                    className="mt-2",
                                ),
                            ]
                        ),
                    ],
                    style=INNER_CARD_STYLE,
                ),
            ],
            style=OUTER_CARD_STYLE,
        ),
        dbc.Card(id="card-featurization-download", children=[
            dbc.Card([
                dbc.CardHeader("Download featurization results", style=HEADER_STYLE),
                dbc.CardBody(children=[
                    html.Div([
                        dbc.Select(
                            id="dropdown-featurization-dataset",
                            options=[],
                            value=None,
                            placeholder="Select a featurization dataset",
                            persistence=True,
                            persistence_type="memory",
                            style={**DROPDOWN_STYLE, "width": "100%"},
                        ),
                        html.Br(),
                        html.Div(children=[
                            dcc.Input(id="input-dwl-featurization-filename", 
                                    placeholder="Enter a filename", 
                                    type="text", 
                                    style={**INPUT_STYLE, "flex": "1"}, 
                                    className="mb-3"),
                            dcc.Markdown('''.csv''', style={"margin": 0}),
                            dbc.Button("Download featurization", 
                                    id="btn-download-featurization", 
                                    style=PRIM_BUTTON_STYLE, 
                                    disabled=True,
                                    className="mb-3"),
                        ], className="d-flex align-items-center gap-2"),
                        dcc.Download(id="download-featurization"),
                        html.Div(id="feedback-download-featurization"),
                    ],style={"maxWidth": "700px", "width": "100%", "margin": "0 auto"})
                ]),
            ], style=INNER_CARD_STYLE),
        ], style={**OUTER_CARD_STYLE, "display": "none"}),
        html.Div(
            id="container-go-to-searchspace-from-featurization",
            style={"display": "none", "textAlign": "center"},
            children=dbc.Button(
                "Continue to search space creation",
                id="btn-go-to-searchspace-from-featurization",
                color="primary",
                className="mt-2",
            ),
        ),
        html.Br(),
    ]),


    # Preprocess page
    html.Div(id='page-preprocess', style={'display': 'none'}, children=[
        dbc.Card([
            dbc.Card([
                dbc.CardHeader(
                    html.Div(
                        [
                            html.Span("Select data for search space creation"),
                            dbc.Button(
                                "< Back to featurization",
                                id="btn-back-to-featurization-from-preprocess",
                                color="light",
                                outline=False,
                                size="sm",
                                className="py-0",
                                style={"whiteSpace": "nowrap"},
                            ),
                        ],
                        className="d-flex justify-content-between align-items-center",
                    ),
                    style=HEADER_STYLE,
                ),
                dbc.CardBody(children=[
                    dcc.Upload(
                        id='upload-preprocess-features',
                        multiple=True,
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select additional feature CSV files')
                        ]),
                        style={**UPLOAD_STYLE, "maxWidth": "700px", "margin": "0 auto"},
                    ),
                    html.Div(id="feedback-preprocess-upload", className="mt-2"),
                    html.Div(id="feature-summary"),  # placeholder for dataset summary table
                ]),
            ],style=INNER_CARD_STYLE),
        ],style=OUTER_CARD_STYLE),
        dbc.Card(id="card-preprocess-download", children=[
            dbc.Card([
                dbc.CardHeader("Create and download search space", style=HEADER_STYLE),
                dbc.CardBody(children=[
                    html.Div([
                        html.Div(children=[
                            dcc.Input(id="input-dwl-preprocess-filename", 
                                    placeholder="Enter a filename", 
                                    type="text", 
                                    style={**INPUT_STYLE, "flex": "1"}, 
                                    className="mb-3"),
                            dcc.Markdown('''.csv''', style={"margin": 0}),
                            dbc.Button("Create and download search space", 
                                    id="btn-create-search-space", 
                                    style={**PRIM_BUTTON_STYLE, "maxWidth": "fit-content", "alignItems": "center"},
                                    className="mb-3",),
                        ], className="d-flex align-items-center gap-2"),
                        dcc.Download(id="download-preprocess"),  # placeholder for download component
                        html.Div(id="feedback-download-preprocess"),  # placeholder for download feedback
                        html.Div(id="feedback-preprocess-info"),  # placeholder for additional info about the search space creation
                    ],style={"maxWidth": "800px", "width": "100%", "margin": "0 auto"}),
                    html.Div(id="table-preprocess-view", 
                             style={"maxWidth": "1200px", 
                                    "width": "100%", 
                                    "margin": 
                                    "0 auto"}
                    )  # placeholder for search space preview table
                ]),
            ], style=INNER_CARD_STYLE),
        ], style={**OUTER_CARD_STYLE, "display": "none"}),
        html.Br(),
    ]),


    # Main functionality page
    html.Div(id='page-main', style={'display': 'none'}, children=[
        dbc.Card(
            [
                dbc.CardHeader(
                    html.Div(
                        [
                            dbc.Tabs([
                                dbc.Tab(label="Substrate selections", tab_id="tab-select", tab_style=MAIN_TAB_STYLE, active_tab_style=MAIN_TAB_ACTIVE_STYLE, label_style=MAIN_TAB_LABEL_STYLE, active_label_style=MAIN_TAB_ACTIVE_LABEL_STYLE),
                                dbc.Tab(label="Visualize scope", tab_id="tab-umap", tab_style=MAIN_TAB_STYLE, active_tab_style=MAIN_TAB_ACTIVE_STYLE, label_style=MAIN_TAB_LABEL_STYLE, active_label_style=MAIN_TAB_ACTIVE_LABEL_STYLE),
                                dbc.Tab(label="Analyze features", tab_id="tab-inference", tab_style=MAIN_TAB_STYLE, active_tab_style=MAIN_TAB_ACTIVE_STYLE, label_style=MAIN_TAB_LABEL_STYLE, active_label_style=MAIN_TAB_ACTIVE_LABEL_STYLE),
                                dbc.Tab(label="Predictive modeling", tab_id="tab-modeling", tab_style=MAIN_TAB_STYLE, active_tab_style=MAIN_TAB_ACTIVE_STYLE, label_style=MAIN_TAB_LABEL_STYLE, active_label_style=MAIN_TAB_ACTIVE_LABEL_STYLE)
                            ], id="tabs-main", 
                            active_tab="tab-select",
                            style={"backgroundColor": background_color}
                            ),
                            dbc.Button(
                                "< Back to starting page",
                                id="btn-back-to-upload-from-main",
                                style={"backgroundColor": back_button_color, "borderColor": back_button_color},
                                size="sm",
                                className="ms-3 align-self-start",
                            ),
                        ],
                        style={"display": "flex", "alignItems": "flex-start", "justifyContent": "space-between", "gap": "0.75rem"},
                    ),
                    style={"backgroundColor": background_color, "border": "none"},
                ),
                dbc.CardBody(
                    html.Div(id="tab-content-main", className="card-text"),
                    style={"minHeight": "400px", "border": "none"},
                ),
            ], style={"borderColor": border_color, "borderWidth": "1px", "borderRadius": "0.25rem", "boxShadow": "0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)"}
        ),
        html.Br()
    ])

], fluid=True)


############################################################
# CALLBACK SECTION
############################################################

####################################
# callback for main page rendering
####################################

@callback(
    Output("tab-content-main", "children"),
    Input("tabs-main", "active_tab"),
    State("store-umap-figure", "data"),
    State("store-shap-beeswarm-figure", "data"),
    State("store-shap-bar-figure", "data"),
    State("store-pred-figure", "data"),
)
def render_tab_content(active_tab, stored_umap_figure, stored_shap_beeswarm_figure, stored_shap_bar_figure, stored_pred_figure):
    if active_tab == "tab-select":
        return html.Div([                
                dbc.Card([
                    dbc.Card([
                        dbc.CardHeader("Select substrates with ScopeBO", style=HEADER_STYLE),
                        dbc.CardBody(children=[
                            html.Div(
                                [
                                    dbc.Button("Run ScopeBO", id="btn-scopebo", style=PRIM_BUTTON_STYLE, className="mb-3", disabled=False),
                                    dcc.Loading(
                                        id="loading-scopebo-run",
                                        type="default",
                                        children=html.Div(id="feedback-scopebo-run", className="mt-2"),
                                    ),
                                ],
                                style={"display": "flex", "flexDirection": "column", "alignItems": "center"},
                            ),
                        ],
                        className="mx-auto mb-4",
                        style={"width": "fit-content"}
                        ),
                    ],style=INNER_CARD_STYLE),
                ],style=OUTER_CARD_STYLE),
                dbc.Card([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Div(
                                [
                                    html.Span("View suggestions and report results"),
                                    dbc.Button(
                                        "?",
                                        id="open-suggestions-help",
                                        size="sm",
                                        style=MODAL_STYLE,
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            ),
                            style=HEADER_STYLE,
                        ),
                        dbc.CardBody(children=[
                            html.Div(
                                dbc.RadioItems(
                                    id='radio-sugg',
                                    options=[
                                        {'label': 'Suggestions', 'value': 'sugg'},
                                        {'label': 'Alternative suggestions', 'value': 'alt'},
                                        {'label': 'Report other substrates', 'value': 'other'},
                                        {'label': 'Modify scope', 'value': 'scope'}
                                    ],
                                    value='sugg',
                                    inline=True,
                                    className="scope-mode-switch btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="scope-mode-option btn",
                                    labelCheckedClassName="active"
                                ),
                                className="scope-mode-switch-wrap mb-3",
                            ),
                            html.Div(id="info-scope", className="mb-2"),
                            html.Div(id="display-table"),  # placeholder for substrate table display
                            html.Div(
                                [
                                    dbc.Button(
                                        "Report these results",
                                        id="btn-report-results",
                                        style=PRIM_BUTTON_STYLE,
                                        className="mt-3",
                                    ),
                                    html.Div(
                                        id="feedback-report-results",
                                        className="mt-2",
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                },
                            ),
                        ]),
                    ], style=INNER_CARD_STYLE),
                ],style=OUTER_CARD_STYLE),
                dbc.Card([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Div(
                                [
                                    html.Span("Download the current search space"),
                                    dbc.Button(
                                        "?",
                                        id="open-scopebo-dwl-help",
                                        size="sm",
                                        style=MODAL_STYLE,
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            ),
                            style=HEADER_STYLE,
                        ),
                        dbc.CardBody(children=[
                            html.Div(children=[
                                dcc.Input(id="input-dwl-filename", placeholder="filename", type="text", className="mb-3"),
                                dcc.Markdown('''.csv'''),
                                dbc.Button("Download search space", id="btn-download-space", style=PRIM_BUTTON_STYLE, className="mb-3",disabled=True),
                            ], className="d-flex justify-content-center align-items-center gap-2"),
                            dcc.Download(id="download-search-space"),
                            html.Div(id="feedback-download", className="mt-2"),
                        ],
                        className="mx-auto mb-4",
                        style={"width": "fit-context"}
                        ),
                    ],style=INNER_CARD_STYLE
                    ),
                ],style=OUTER_CARD_STYLE),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Searchspace download help")),
                        dbc.ModalBody(
                            [
                                dcc.Markdown("You can download the current search space here."),
                                dcc.Markdown("The priority column will indicate the sample status:"),
                                dbc.Table(
                                    [
                                        html.Thead(
                                            html.Tr([
                                                html.Th("Priority"),
                                                html.Th("Status"),
                                            ])
                                        ),
                                        html.Tbody(
                                            [
                                                html.Tr([
                                                    html.Td("1"),
                                                    html.Td("Suggested substrate"),
                                                ]),
                                                html.Tr([
                                                    html.Td("0 < priority < 1"),
                                                    html.Td("Alternative suggestion (higher value = higher priority)"),
                                                ]),
                                                html.Tr([
                                                    html.Td("0"),
                                                    html.Td("Unseen substrate"),
                                                ]),
                                                html.Tr([
                                                    html.Td("-1"),
                                                    html.Td("Pruned substrate"),
                                                ]),
                                                html.Tr([
                                                    html.Td("-2"),
                                                    html.Td("Substrate with experimental results (scope substrate)"),
                                                ]),
                                            ]
                                        ),
                                    ],
                                    bordered=True,
                                    hover=True,
                                    responsive=True,
                                    striped=True,
                                    size="sm",
                                )
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-scopebo-dwl-help",
                                color="secondary",
                            )
                        ),
                    ],
                    id="scopebo-dwl-help-modal",
                    className="help-modal",
                    is_open=False,
                    dialog_style={"width": "90%","maxWidth": "1200px"},
                    centered=True,
                    scrollable=True,
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Reporting help")),
                        dbc.ModalBody(
                            [
                                dcc.Markdown("You can report results in this section."),
                                dcc.Markdown("- Fill in experimental results in the objective columns and click 'Report these results'."),
                                dcc.Markdown("- Per round, ScopeBO provides three suggested substrates and five alternatives."),
                                dcc.Markdown("- Use the three suggestions if possible and only use the alternatives if the suggestions are not available."),
                                dcc.Markdown("- **First round suggestions are random. We recommend instead using three substrates with different functional groups that are predicted to work well in the reaction based on chemical intuition.**"),
                                dcc.Markdown("- You can report such other results under 'Report other substrates'."),
                                dcc.Markdown("- Under 'Modify scope', you can update reported scope results."),
                                dcc.Markdown("- Results highlighted in green are up to date with the stored search space."),
                                dcc.Markdown("- Results highlighted in yellow have not been saved yet."),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-suggestions-help",
                                color="secondary",
                            )
                        ),
                    ],
                    id="suggestions-help-modal",
                    className="help-modal",
                    is_open=False,
                    dialog_style={"width": "90%","maxWidth": "1200px"},
                    centered=True,
                    scrollable=True,
                )
            ])
    elif active_tab == "tab-umap":
        return html.Div([
            dbc.Card([
                dbc.Card([
                    dbc.CardHeader("Chemical space visualization", style=HEADER_STYLE),
                    dbc.CardBody(children=[
                        html.Div(
                            [
                                dbc.Select(
                                    id="umap-objective-dropdown",
                                    options=[
                                        {
                                            "label": "Select objective to visualize",
                                            "value": "Select objective to visualize",
                                        }
                                    ],
                                    value="Select objective to visualize",
                                    persistence=True,
                                    persistence_type="memory",
                                    style={**DROPDOWN_STYLE, "maxWidth": "300px", "margin": "0 auto"},
                                ),
                                dbc.Button(
                                    "Create or update visualization",
                                    id="btn-refresh-umap",
                                    style={**PRIM_BUTTON_STYLE, "width": "300px", "margin": "0 auto"},
                                    className="ms-2",
                                    disabled=True,
                                ),
                            ],
                            className="d-flex justify-content-center align-items-center mb-3",
                            style = {"width": "fit-content", "margin": "0 auto"}
                        ),
                        html.Div(id="alert-umap-stale", className="mb-2"),
                        dcc.Loading(
                            id="loading-umap-visualization",
                            type="default",
                            children=html.Div(
                                html.Div(
                                    id="visual-umap",
                                    children=_build_umap_graph_children(stored_umap_figure),
                                    style={"width": "100%", "max-width": "1000px"}
                                ),  # placeholder for UMAP
                                className="d-flex justify-content-center",
                            ),
                        ),
                    ],
                    className="mx-auto mb-4",
                    style={"width": "100%"}
                    ),
                ],style=INNER_CARD_STYLE),
            ],style=OUTER_CARD_STYLE),
            html.Br(),
            dbc.Card([
                dbc.Card([
                    dbc.CardHeader("Scope summary", style=HEADER_STYLE),
                    dbc.CardBody(children=[
                        html.Div(id="scope-visualization", className="mt-2"),  # placeholder for RDKit picture of full scope
                    ],
                    className="mx-auto mb-4",
                    style={"width": "100%"}
                    ),
                ],style=INNER_CARD_STYLE),
            ],style=OUTER_CARD_STYLE),
            html.Br(),
        ])
    elif active_tab == "tab-inference":
        return html.Div([
            dbc.Card([
                dbc.Card([
                    dbc.CardHeader("SHAP analysis", style=HEADER_STYLE),
                    dbc.CardBody(children=[
                        html.Div(
                            [
                                dbc.Select(
                                    id="shap-objective-dropdown",
                                    options=[
                                        {
                                            "label": "Select objective to analyze",
                                            "value": "Select objective to analyze",
                                        }
                                    ],
                                    value="Select objective to analyze",
                                    persistence=True,
                                    persistence_type="memory",
                                    style={**DROPDOWN_STYLE, "maxWidth": "300px", "margin": "0 auto"},
                                ),
                                dbc.Button(
                                    "Create or update analysis",
                                    id="btn-refresh-shap",
                                    style={**PRIM_BUTTON_STYLE, "width": "300px", "margin": "0 auto"},
                                    className="ms-2",
                                ),
                            ],
                            className="d-flex justify-content-center align-items-center mb-3",
                            style = {"width": "fit-content", "margin": "0 auto"}
                        ),
                        html.Div(id="alert-shap-stale", className="mb-2"),
                        dcc.Loading(
                            id="loading-shap",
                            type="default",
                            children=html.Div(
                                id="visual-shap",
                                className="mt-2",
                                children=_build_shap_graph_children(
                                    stored_shap_beeswarm_figure,
                                    stored_shap_bar_figure,
                                ),
                            ),  # placeholder for SHAP
                        ),
                    ]),
                ], style=INNER_CARD_STYLE),
            ], style=OUTER_CARD_STYLE),
            html.Br(),
        ])
    elif active_tab == "tab-modeling":
        return html.Div([
            dbc.Card([
                dbc.Card([
                    dbc.CardHeader("Generate and view predictions", style=HEADER_STYLE),
                    dbc.CardBody(children=[
                        html.Div(
                            [
                                dbc.Select(
                                    id="pred-objective-dropdown",
                                    options=[
                                        {
                                            "label": "Select objective to predict",
                                            "value": "Select objective to predict",
                                        }
                                    ],
                                    value="Select objective to predict",
                                    persistence=True,
                                    persistence_type="memory",
                                    style={**DROPDOWN_STYLE, "maxWidth": "300px", "margin": "0 auto"},
                                ),
                                dbc.Button(
                                    "Create or update predictions",
                                    id="btn-refresh-pred",
                                    style={**PRIM_BUTTON_STYLE, "width": "300px", "margin": "0 auto"},
                                    className="ms-2",
                                ),
                            ],
                            className="d-flex justify-content-center align-items-center mb-3",
                            style = {"width": "fit-content", "margin": "0 auto"}
                        ),
                        html.Div(id="alert-pred-stale", className="mb-2"),
                        dcc.Loading(
                            id="loading-pred",
                            type="default",
                            children=html.Div(
                                id="visual-pred",
                                className="mt-2",
                                children=_build_pred_graph_children(stored_pred_figure),
                            ),  # placeholder for predictions
                        ),
                    ]),
                ], style=INNER_CARD_STYLE),
            ], style=OUTER_CARD_STYLE),
            html.Br(),
            dbc.Card(id="card-pred-download",children=[
                dbc.Card(
                    children=[
                        dbc.CardHeader("Download predictions", style=HEADER_STYLE),
                        dbc.CardBody(children=[
                            dcc.Markdown('''#### Download predictions''', className="text-center mb-4"),
                            html.Div(children=[
                                dcc.Input(id="input-dwl-pred-filename", placeholder="filename", type="text", className="mb-3"),
                                dcc.Markdown('''.csv'''),
                                dbc.Button("Download predictions", id="btn-download-pred", style=PRIM_BUTTON_STYLE, className="mb-3", disabled=True),
                            ], className="d-flex justify-content-center align-items-center gap-2"),
                            dcc.Download(id="download-predictions"),
                            html.Div(id="feedback-download-pred", className="mt-2"),
                        ]),
                    ],
                style=INNER_CARD_STYLE,
                ),
            ], style={**OUTER_CARD_STYLE, "display": "none"}),
            html.Br(),
        ])
    

@callback(
    Output('page-main', 'style', allow_duplicate=True),
    Output('page-upload', 'style', allow_duplicate=True),
    Input('btn-back-to-upload-from-main', 'n_clicks'),
    prevent_initial_call=True,
)
def go_back_to_upload_from_main(n_clicks):
    """Navigate from main page back to upload page."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


####################################
# callbacks for starting page
####################################

@callback(
    Output("searchspace-help-modal", "is_open"),
    Input("open-searchspace-help", "n_clicks"),
    Input("close-searchspace-help", "n_clicks"),
    State("searchspace-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal_searchspace(open_clicks, close_clicks, is_open):
    """Show or hide the search space help modal."""
    return not is_open


@callback(
    Output("objectives-help-modal", "is_open"),
    Input("open-objectives-help", "n_clicks"),
    Input("close-objectives-help", "n_clicks"),
    State("objectives-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal_objectives(open_clicks, close_clicks, is_open):
    """Show or hide the objectives help modal."""
    return not is_open


@callback(
    Output("feat-overview-help-modal", "is_open"),
    Input("open-feat-overview-help", "n_clicks"),
    Input("close-feat-overview-help", "n_clicks"),
    State("feat-overview-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal_feat_overview(open_clicks, close_clicks, is_open):
    """Show or hide the featurization overview help modal."""
    return not is_open


@callback(
    Output("store-search-space", "data"),
    Output('feedback-searchspace-upload', 'children'),
    Output('feedback-obj-inference', 'children'),
    Output("btn-search-to-main", "disabled"),
    Output("store-objectives", "data"),
    Input("upload-searchspace", "contents"),
    State("upload-searchspace", "filename"),
    prevent_initial_call=True,
)
def store_search_space(contents, filename):
    """Upload and store the search space file, and provide feedback on the upload status."""

    if contents is None:
        return None, no_update, no_update, True, no_update
    
    obj_msg = ""  # no message if no objectives were found in the file
    
    if not filename.lower().endswith('.csv'):  # check for correct file type
        status_msg = dbc.Alert(f"Invalid file type: {filename}. Please upload a CSV file.", 
                               style=DANGER_ALERT_STYLE, 
                               dismissable=False,)
        return None, status_msg, obj_msg, True, no_update  # continue button remains disabled
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_search = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0, header=0, low_memory=False)
    df_search = _drop_fully_empty_rows(df_search)

    # check for minimum search space size
    if len(df_search) < 58:
        status_msg = dbc.Alert(
            f"Search space is too small: {len(df_search)} entries found. Please upload a search space with at least 58 entries.",
            style=DANGER_ALERT_STYLE,
            dismissable=False)
        return None, status_msg, obj_msg, True, no_update  # continue button remains disabled
    
    objectives, found_objs = _infer_obj_from_space(df_search)  # try to infer the objectives from the df

    # check if the index contains valid SMILES strings
    invalid = [
        smi
        for smi in df_search.index.astype(str)
        if Chem.MolFromSmiles(smi) is None
    ]
    if invalid:
        return (
            None,
            dbc.Alert(
                f"{filename}: Index contains invalid SMILES "
                f"(first invalid: '{invalid[0]}').",
                style=DANGER_ALERT_STYLE,
                dismissable=False,
            ), obj_msg, True, no_update
        )
    
    # adjust the phrasing based on the number of objectives that were found
    if found_objs:
        obj_string = "objective"
        obj_phrasing = "this is"
        if len(objectives) > 1:
            obj_string = "objectives"
            obj_phrasing = "these are"
        obj_msg = dbc.Alert(
            [f"Identified the following {obj_string} in the search space file: {', '.join(objectives)}.",
            html.Br(),
            f" Please verify and modify the optimization mode if needed."],
            style=INFO_ALERT_STYLE,
            dismissable=False)
        
    obj_data = [{"name": obj, "mode": "max"} for obj in objectives]  # default to "max" mode for all objectives

    status_msg = dbc.Alert(f"Successfully uploaded file: {filename}", style=SUCCESS_ALERT_STYLE, dismissable=False)

    return df_search.to_json(date_format='iso', orient='split'), status_msg, obj_msg, False, obj_data  # enable continue button


@callback(
    Output("store-objectives", "data", allow_duplicate=True),
    Input("btn-add-obj", "n_clicks"),
    Input("btn-remove-obj", "n_clicks"),
    State("store-objectives", "data"),
    prevent_initial_call=True,
)
def modify_obj_fields(add_clicks, remove_clicks, values):
    """ Adds or removes objective input fields based on which button was clicked."""
    trigger = ctx.triggered_id
    values = values or [{"name": "", "mode": "max"}]  # initialize with one empty objective if no objectives exist yet
    if trigger == "btn-add-obj":
        return values + [{"name": "", "mode": "max"}]
    if trigger == "btn-remove-obj":
        return values[:-1] if len(values) > 1 else values
    return values


@callback(
    Output("store-objectives", "data", allow_duplicate=True),
    Input({"type": "dynamic-input", "index": ALL}, "value"),
    Input({"type": "dynamic-mode", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_values(names, modes):
    """ Saves the current values of all dynamic input fields to the store."""
    # Preserve all current input values
    return [{"name": name, "mode": mode} for name, mode in zip(names, modes)]


@callback(
    Output("input-objectives", "children"),
    Input("store-objectives", "data"),
)
def render_fields(values):
    """Renders the dynamic objective input fields based on the current values in the store."""
    values = values or [{"name": "", "mode": "max"}]
    return [
    html.Div(
        [
            dbc.Input(
                id={"type": "dynamic-input", "index": i},
                value=item.get("name", ""),
                placeholder=f"Objective {i + 1}",
                style={**INPUT_STYLE, "width": "150px"},
            ),

            dbc.Select(
                id={"type": "dynamic-mode", "index": i},
                value=item.get("mode", "max"),
                options=[
                    {"label": "This objective will be maximized.", "value": "max"},
                    {"label": "This objective will be minimized.", "value": "min"},
                ],
                style={**DROPDOWN_STYLE, "maxWidth": "300px"},
            ),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "alignItems": "center",
            "marginBottom": "10px",
        },
    )
    for i, item in enumerate(values)
    ]


@callback(
    Output('page-upload', 'style', allow_duplicate=True),
    Output('page-featurization', 'style'),
    Input('btn-go-to-features', 'n_clicks'),
    prevent_initial_call=True,
)
def go_to_featurization(n_clicks):
    """Navigate from upload page to search-space creation selection page."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


@callback(
    Output('page-upload', 'style', allow_duplicate=True),
    Output('page-preprocess', 'style', allow_duplicate=True),
    Input('btn-go-to-preprocess', 'n_clicks'),
    prevent_initial_call=True,
)
def go_to_preprocess_from_upload(n_clicks):
    """Navigate from upload page (starting) to preprocess page."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


@callback(
    Output('page-upload', 'style'),
    Output('page-main', 'style'),
    Output('store-search-space', 'data',allow_duplicate=True),
    Output('store-init', 'data'),
    Output("feedback-obj-status", "children"),
    Input('btn-search-to-main', 'n_clicks'),
    State('store-search-space', 'data'),
    State('store-objectives', 'data'),
    running=[(Output("btn-search-to-main", "disabled"), True, False)],
    prevent_initial_call=True,
)
def go_to_main(n_clicks, search_space, objective_data):
    """
    Move from the search space upload page to the main page. 
    Also check the search space to ensure proper formatting.
    Store data about experimental status to inform main page layout.
    """
    # check for objective fields without names and prevent moving forward if any are found
    objective_data = objective_data or [{"name": "", "mode": "max"}]
    objectives = [obj["name"] for obj in objective_data]
    if objectives:
        for obj in objectives:
            if not obj.strip():  # check for empty or whitespace-only objective names
                return no_update, no_update, no_update, no_update, dbc.Alert(
                    "Please fill in a name for all objectives before proceeding.",
                    style=DANGER_ALERT_STYLE,
                    dismissable=False)

    if n_clicks and search_space:
        search_space = pd.read_json(search_space, orient='split')
        updated_search_space = _ensure_search_space_layout(search_space, objectives)
        # Parse high-level scope state for UI flow and initialization logic.
        _, _, status_init = _parse_space(updated_search_space)
        updated_search_space_json = updated_search_space.to_json(date_format='iso', orient='split')
        return {'display': 'none'}, {'display': 'block'}, updated_search_space_json, status_init, no_update
    return no_update, no_update, no_update, no_update, no_update


####################################
# callbacks for substrate suggestion
####################################

@callback(
    Output("suggestions-help-modal", "is_open"),
    Input("open-suggestions-help", "n_clicks"),
    Input("close-suggestions-help", "n_clicks"),
    State("suggestions-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal_suggestions(open_clicks, close_clicks, is_open):
    """Show or hide the suggestions help modal."""
    return not is_open


@callback(
    Output("scopebo-dwl-help-modal", "is_open"),
    Input("open-scopebo-dwl-help", "n_clicks"),
    Input("close-scopebo-dwl-help", "n_clicks"),
    State("scopebo-dwl-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal_scopebo_dwl(open_clicks, close_clicks, is_open):
    """Show or hide the ScopeBO download help modal."""
    return not is_open


@callback(
    Output("btn-download-space", "disabled"),
    Input("store-search-space", "data"),
)
def toggle_download_button(search_space):
    """Enable download only when a search space exists in the store."""
    return not bool(search_space)


@callback(
    Output("download-search-space", "data"),
    Output("feedback-download", "children"),
    Input("btn-download-space", "n_clicks"),
    State("store-search-space", "data"),
    State("input-dwl-filename", "value"),
    prevent_initial_call=True,
)
def download_search_space(n_clicks, search_space, filename):
    """Download the current search space as a CSV file."""
    if not n_clicks:
        return no_update, no_update

    if not search_space:
        return no_update, dbc.Alert(
            "No search space available to download yet.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    df_search = pd.read_json(search_space, orient="split")

    raw_name = (filename or "").strip()
    if raw_name:
        safe_name = os.path.basename(raw_name).replace("\\", "_").replace("/", "_")
    else:
        safe_name = f"scopebo_search_space_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    return dcc.send_data_frame(df_search.to_csv, safe_name, index=True), dbc.Alert(
        f"Download successful: {safe_name}",
        style=SUCCESS_ALERT_STYLE,
        dismissable=False,
    )


@callback(
    Output("store-other-rows", "data"),
    Input("btn-add-other-row", "n_clicks"),
    State("store-other-rows", "data"),
    prevent_initial_call=True,
)
def add_other_row(n_clicks, other_rows):
    """Append a blank row to the 'other substrates' table."""
    if not n_clicks:
        return no_update

    other_rows = other_rows or []
    next_row_id = max((row.get("row_id", -1) for row in other_rows), default=-1) + 1
    return [*other_rows, {"row_id": next_row_id, "smiles": ""}]


@callback(
    Output("display-table", "children"),
    Input("radio-sugg", "value"),
    Input("store-search-space", "data"),
    Input("store-other-rows", "data"),
    State("store-objectives", "data"),
)
def render_reporting_table(radio_value, search_space, other_rows, objectives_dict):
    """Renders the appropriate substrate reporting table based on the selected radio button value."""
    objectives = [obj["name"] for obj in objectives_dict]  # extract objective names from the stored data
    other_rows = other_rows or []

    if not search_space:
        return dbc.Alert("No search space loaded yet. Please upload a search space first.", style=WARNING_ALERT_STYLE)
    
    # Always derive visible rows from current search-space state (single source of truth).
    df_search = pd.read_json(search_space, orient='split')

    if not "priority" in df_search.columns:
        return no_update

    list_sugg = df_search.index[df_search['priority'] == 1].to_list()
    list_alt = df_search.index[(df_search['priority'] < 1) & (df_search['priority'] > 0)].to_list()

    # "Existing scope" = compounds where all objective values are already known.
    if objectives:
        list_scope = df_search.index[~df_search[objectives].astype(str).eq("PENDING").any(axis=1)].to_list()
    else:
        list_scope = df_search.index.to_list()
    
    if radio_value == "sugg":
        if list_sugg:
            return _build_table(list_sugg, objectives, df_search)
        else:
            return dbc.Alert(
                "No suggestions available yet. Please run ScopeBO to generate suggestions.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            )
    elif radio_value == "alt":
        if list_alt:
            return _build_table(list_alt, objectives, df_search)
        else:
            return dbc.Alert(
                "No alternative suggestions available yet. Please run ScopeBO to generate suggestions.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            )
    elif radio_value == "scope":
        if list_scope:
            return _build_table(list_scope, objectives, df_search)
        else:
            return dbc.Alert(
                "No scope compounds with complete objective values are available yet.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            )
    elif radio_value == "other":
        return _build_other_table(other_rows, objectives, df_search)


@callback(
    Output("info-scope", "children"),
    Input("radio-sugg", "value"),
    Input("store-search-space", "data"),
    State("store-objectives", "data"),
    State("store-init", "data"),
    State("btn-scopebo", "n_clicks")
)
def render_scope_size_info(radio_value, search_space, objectives_dict, init_status, n_clicks):
    """Show mode-specific info/warnings above the reporting table."""
    
    if (radio_value == "sugg") and init_status and n_clicks:
        return dbc.Alert(
            "These initial sugggestions have been recommended randomly. For scope initiation,"\
                " we recommend using three diverse compounds with good reactivity.\nYou can report "\
                    "these in the 'Report other substrates' mode.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )
    
    if (radio_value == "alt") and init_status and n_clicks:
        return dbc.Alert(
            "As initial suggestions are generated randomly, no alternative suggestions were generated.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )
    
    if radio_value == "other" and not init_status and n_clicks:
        return dbc.Alert(
            "Apart from scope initiation, we recommend to use the suggestions provided by ScopeBO for your scope.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )
    
    if radio_value == "alt" and not init_status and n_clicks:
        return dbc.Alert(
            "Use one of these compounds as an alternative to the main suggestions only if you have a specific reason to do so.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    if radio_value != "scope" or not search_space:
        return None

    objectives = [obj["name"] for obj in objectives_dict]
    df_search = pd.read_json(search_space, orient='split')

    if objectives:
        list_scope = df_search.index[~df_search[objectives].astype(str).eq("PENDING").any(axis=1)].to_list()
    else:
        list_scope = df_search.index.to_list()

    return dbc.Alert(
        f"Current scope size: {len(list_scope)} compounds.",
        style=INFO_ALERT_STYLE,
        dismissable=False,
    )


@callback(
    Output("store-other-rows", "data", allow_duplicate=True),
    Input({"type": "other-smiles-input", "row_id": ALL}, "value"),
    State({"type": "other-smiles-input", "row_id": ALL}, "id"),
    State("store-other-rows", "data"),
    prevent_initial_call=True,
)
def sync_other_smiles_rows(smiles_values, smiles_ids, other_rows):
    """Persist editable SMILES values for the 'other substrates' table."""
    other_rows = other_rows or []
    rows_by_id = {row.get("row_id"): dict(row) for row in other_rows}

    for component_id, value in zip(smiles_ids, smiles_values):
        row_id = component_id.get("row_id")
        if row_id in rows_by_id:
            rows_by_id[row_id]["smiles"] = (value or "").strip()

    return list(rows_by_id.values())


@callback(
    Output("store-search-space", "data", allow_duplicate=True),
    Output("store-updated", "data", allow_duplicate=True),
    Output("feedback-report-results", "children", allow_duplicate=True),
    Input("btn-report-results", "n_clicks"),
    State("radio-sugg", "value"),
    State("store-search-space", "data"),
    State("store-objectives", "data"),
    State({"type": "objective-input", "smiles": ALL, "objective": ALL}, "id"),
    State({"type": "objective-input", "smiles": ALL, "objective": ALL}, "value"),
    State("store-other-rows", "data"),
    State({"type": "other-objective-input", "row_id": ALL, "objective": ALL}, "id"),
    State({"type": "other-objective-input", "row_id": ALL, "objective": ALL}, "value"),
    prevent_initial_call=True,
)
def report_results(n_clicks, radio_value, search_space, objectives_dict, input_ids, input_values, other_rows, other_input_ids, other_input_values):
    """Updates the search space with entered results when the 'Report these results' button is clicked."""
    
    if not n_clicks or not search_space:
        return no_update, no_update, no_update

    objective_names = [obj["name"] for obj in objectives_dict]
    df_search = pd.read_json(search_space, orient='split')
    df_search = _ensure_search_space_layout(df_search, objective_names)
    missing_smiles_count = 0

    if radio_value == "other":
        updated_df, updated_rows, skipped_rows, invalid_numeric_count, missing_smiles_count = _update_search_space_from_other_report(
            df_search,
            other_rows,
            other_input_ids,
            other_input_values,
            objective_names,
        )
    else:
        updated_df, updated_rows, skipped_rows, invalid_numeric_count = _update_search_space_from_report(
            df_search,
            input_ids,
            input_values,
            objective_names,
        )

    # Build independent success/warning alerts so users can see both outcomes clearly.
    alerts = []

    if updated_rows > 0:
        alerts.append(dbc.Alert(
            f"Updated {updated_rows} substrates in the search space.",
            style=SUCCESS_ALERT_STYLE,
            dismissable=False,
        ))

    warning_feedback = []
    if skipped_rows:
        warning_feedback.append(f"Skipped {skipped_rows} incomplete substrate(s).")
    if invalid_numeric_count:
        if warning_feedback:
            warning_feedback.append(html.Br())
        warning_feedback.append(
            f"Ignored {invalid_numeric_count} non-numeric value(s). Please enter numbers only."
        )
    if missing_smiles_count:
        if warning_feedback:
            warning_feedback.append(html.Br())
        warning_feedback.append(
            f"Ignored {missing_smiles_count} substrate(s) because their SMILES are not present in the search space."
        )

    if warning_feedback:
        alerts.append(dbc.Alert(
            warning_feedback,
            style=WARNING_ALERT_STYLE,
            dismissable=False,
            className="mt-2" if updated_rows > 0 else None,
        ))

    if updated_rows == 0 and not warning_feedback:
        alerts.append(dbc.Alert(
            "No changes detected. Reported values match the existing values in the search space.",
            style=INFO_ALERT_STYLE,
            dismissable=False,
        ))

    if updated_rows > 0:
        return updated_df.to_json(date_format='iso', orient='split'), True, alerts
    return no_update, no_update, alerts if alerts else no_update


@callback(
    Output("feedback-report-results", "children"),
    Input("radio-sugg", "value"),
    Input("btn-scopebo", "n_clicks"),
    prevent_initial_call=True,
)
def clear_report_feedback(_, __):
    """Clear report feedback when switching between suggestion table views or when running ScopeBO."""
    return None


@callback(
    Output("feedback-scopebo-run", "children", allow_duplicate=True),
    Input("btn-report-results", "n_clicks"),
    prevent_initial_call=True,
)
def clear_scopebo_feedback_on_report(_):
    """Clear ScopeBO run feedback after reporting results."""
    return None


@callback(
    Output({"type": "objective-input", "smiles": ALL, "objective": ALL}, "style"),
    Input({"type": "objective-input", "smiles": ALL, "objective": ALL}, "value"),
    State({"type": "objective-input", "smiles": ALL, "objective": ALL}, "id"),
    State("store-search-space", "data"),
)
def style_objective_inputs(input_values, input_ids, search_space):
    """Color inputs live: green if value matches stored value, yellow otherwise."""
    if not input_ids:
        return []

    df_search = pd.read_json(search_space, orient='split') if search_space else None
    styles = []

    for value, field_id in zip(input_values, input_ids):
        # Neutral style if nothing is entered yet.
        if not pd.notna(value) or value == "":
            styles.append({"border": "1px solid #ced4da"})
            continue

        smiles = field_id.get("smiles")
        objective = field_id.get("objective")

        stored_value = None
        if (
            df_search is not None
            and smiles in df_search.index
            and objective in df_search.columns
        ):
            candidate = df_search.loc[smiles, objective]
            if pd.notna(candidate) and str(candidate).upper() != "PENDING":
                stored_value = candidate

        is_match = False
        if stored_value is not None:
            try:
                is_match = float(value) == float(stored_value)
            except (TypeError, ValueError):
                is_match = str(value) == str(stored_value)

        if is_match:
            styles.append({"backgroundColor": "#d1e7dd", "border": "1px solid #198754"})
        else:
            styles.append({"backgroundColor": "#fff3cd", "border": "1px solid #ffc107"})

    return styles


@callback(
    Output("feedback-scopebo-run", "children", allow_duplicate=True),
    Output("store-search-space", "data", allow_duplicate=True),
    Output("store-updated", "data", allow_duplicate=True),
    Output("store-init", "data", allow_duplicate=True),
    Input("btn-scopebo", "n_clicks"),
    State("store-search-space", "data"),
    State("store-objectives", "data"),
    State("store-updated", "data"),
    State("store-init", "data"),
    running=[(Output("btn-scopebo", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_scopebo_button(n_clicks, search_space, obj_dict, updated_flag, init_flag):
    """Runs the ScopeBO algorithm when the button is clicked."""
    if n_clicks:
        search_space_df = pd.read_json(search_space, orient='split')  # read the current search space
        objectives_ = [obj["name"] for obj in obj_dict]
        objectives_modes = {obj["name"]: obj["mode"] for obj in obj_dict}
        
        # If no measured objectives exist yet, remove priority so ScopeBO enters init sampling mode.
        if objectives_:
            has_experimental_results = ~search_space_df[objectives_].astype(str).eq("PENDING").any(axis=1)
            if not has_experimental_results.any() and "priority" in search_space_df.columns:
                search_space_df = search_space_df.drop(columns=["priority"])
        
        if not updated_flag:
            return dbc.Alert(
                "Suggestions were already generated. Please report results below before running ScopeBO again.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            ), no_update, no_update, no_update
        
        # ScopeBO currently expects file input; write a temp CSV for this run.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # timestamp for unique filename
        search_space_df.to_csv(f"scopebo_webapp_run_{timestamp}.csv", index=True)
        try:
            run_df = ScopeBO().run(filename=f"scopebo_webapp_run_{timestamp}.csv", objectives=objectives_, objective_mode=objectives_modes)
            print(run_df[run_df['priority'] > 0], flush=True)
            updated_flag = False  # reset the updated flag after a successful run
        except Exception as e:  # provide feedback on the error if ScopeBO.run() fails
            return _build_traceback_alert("running ScopeBO", e), no_update, no_update, no_update
        
        # delete the temporary CSV file after the run
        os.remove(f"scopebo_webapp_run_{timestamp}.csv")
        
        # Persist updated priorities back into the canonical store.
        updated_search_space_json = run_df.to_json(date_format='iso', orient='split')

        # check if the initialization flag should be set to False (i.e., if there are now experimental results)
        if init_flag and not run_df[objectives_].astype(str).eq("PENDING").any(axis=1).all():
            init_flag = False

        print(f"ScopeBO run completed. Suggestions: {run_df.index[run_df['priority'] == 1].to_list()}", flush=True)
        print(f"Alternatives: {run_df.index[(run_df['priority'] < 1) & (run_df['priority'] > 0)].to_list()}", flush=True)

        # check if the scope already has the full size (27 samples)
        if len(run_df.index[~run_df[objectives_].astype(str).eq("PENDING").any(axis=1)]) >= 27:
            return dbc.Alert(
                "ScopeBO has been run successfully. The optimal scope size of 27 samples has however already been reached.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            ), updated_search_space_json, updated_flag, init_flag
        
        # check if this will be the last round
        if len(run_df.index[~run_df[objectives_].astype(str).eq("PENDING").any(axis=1)]) >= 24:
            return dbc.Alert(
                "ScopeBO has been run successfully. With this round, the optimal scope size of 27 samples will be reached.",
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            ), updated_search_space_json, updated_flag, init_flag


        return dbc.Alert("ScopeBO has been run successfully."\
                         " Suggestions are shown below. ", style=SUCCESS_ALERT_STYLE, dismissable=False), updated_search_space_json, updated_flag, init_flag
    return no_update, no_update, no_update, no_update


####################################
# callbacks for UMAP modeling
####################################

@callback(
    Output("store-scope-grid", "data"),
    Input("store-search-space", "data"),
    State("store-objectives", "data")
)
def generate_scope_grid(search_space, objectives_dict):
    """Shows the full scope as a RDKit image grid with objective values"""

    if not search_space or not objectives_dict:
        return None
    
    search_space = pd.read_json(search_space, orient='split')
    objectives = [obj["name"] for obj in objectives_dict]

    if not "priority" in search_space.columns:
        return no_update

    # get the scope compounds
    scope_compounds = search_space.loc[
        ~search_space[objectives].astype(str).eq("PENDING").any(axis=1)]
    
    if scope_compounds.empty:
        return None
    
    children = []

    for smiles in scope_compounds.index:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # RDKit → PNG (single molecule)
        img = Draw.MolToImage(mol, size=(400, 400))

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode()

        legend = "\n".join(
            f"{obj}: {scope_compounds.loc[smiles, obj]}"
            for obj in objectives
        )

        children.append(
            html.Div(
                [
                    html.Img(
                        src=f"data:image/png;base64,{encoded}",
                        style={
                            "width": "100px",
                            "height": "100px",
                            "display": "block",
                        },
                    ),
                    html.Div(
                        legend,
                        style={
                            "fontSize": "14px",
                            "textAlign": "center",
                            "whiteSpace": "pre-line",
                            "maxWidth": "100px",
                        },
                    ),
                ],
                style={
                    "width": "100px",
                },
            )
        )

    return children

@callback(
    Output("scope-visualization", "children"),
    Input("store-scope-grid", "data")
)
def render_scope_grid(scope_grid_data):
    """Render the scope grid from stored data."""
    if not scope_grid_data:
        return dbc.Alert(
            "No scope data available for visualization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    return html.Div(
        scope_grid_data,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, 100px)",
            "gap": "12px",
            "justifyContent": "center",
            "width": "100%",
        },
    )


@callback(
    Output("umap-objective-dropdown", "options"),
    Output("umap-objective-dropdown", "value"),
    Input("store-objectives", "data"),
    State("umap-objective-dropdown", "value"),
)
def populate_umap_objective_dropdown(objectives_dict, current_value):
    """Populate UMAP objective dropdown from selected objectives."""
    return _objective_dropdown_payload(objectives_dict, current_value, action = "visualize")


@callback(
    Output("btn-refresh-umap", "disabled"),
    Input("umap-objective-dropdown", "value"),
)
def toggle_umap_refresh_button(selected_objective):
    """Enable UMAP refresh button only after objective selection."""
    return (
        selected_objective is None
        or str(selected_objective).startswith("Select objective to")
    )


@callback(
    Output("alert-umap-stale", "children"),
    Input("store-search-space", "data"),
    Input("store-umap-searchspace-signature", "data"),
)
def render_umap_stale_warning(search_space, stored_signature):
    """Warn when the saved UMAP no longer reflects the current search space."""
    return _render_stale_plot_warning(search_space, stored_signature, "visualization")


@callback(
    Output("visual-umap", "children"),
    Output("store-umap-figure", "data"),
    Output("store-umap-searchspace-signature", "data"),
    Input("btn-refresh-umap", "n_clicks"),
    State("store-search-space", "data"),
    State("store-objectives", "data"),
    State("umap-objective-dropdown", "value"),
    prevent_initial_call=True,
)
def umap_visualization(_refresh_clicks, search_space, objectives_dict, selected_objective):
    """Refresh UMAP only when the user clicks the button."""

    if not _refresh_clicks:
        return no_update, no_update, no_update

    if not search_space or not objectives_dict:
        return dbc.Alert(
            "No search space or objectives available for UMAP visualization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update

    # separate the data and calculate UMAP coords using the ScopeBO class
    objectives = [obj["name"] for obj in objectives_dict]
    if not objectives:
        return dbc.Alert(
            "No valid objectives available for UMAP visualization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update

    if selected_objective not in objectives:
        selected_objective = objectives[0]
    objectives_for_visualization = [selected_objective]

    search_space_df = pd.read_json(search_space, orient='split')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # timestamp for unique filename
    temp_filename = f"scopebo_webapp_{timestamp}.csv"
    search_space_df.to_csv(temp_filename, index=True)
    try:
        df_dict = ScopeBO().visualize(
            filename=temp_filename,
            obj_to_show=objectives_for_visualization[0],
            objectives=objectives,
            draw_structures=False,
            show_figure=False,
            return_dfs=True,
        )
    except Exception as e:  # provide feedback on the error if ScopeBO.visualize() fails
        return _build_traceback_alert("building UMAP", e), no_update, no_update
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # Build an interactive UMAP with hover tooltips that render RDKit structures.
    df_seen = df_dict.get("seen", pd.DataFrame()).copy()
    df_suggested = df_dict.get("suggested", pd.DataFrame()).copy()
    df_alternatives = df_dict.get("alternatives", pd.DataFrame()).copy()
    df_neutral = df_dict.get("neutral", pd.DataFrame()).copy()
    df_cut = df_dict.get("cut", pd.DataFrame()).copy()

    # Pre-warm the SMILES image cache once per refresh so hover stays responsive.
    smiles_to_cache = set(df_seen.index.astype(str).tolist())
    smiles_to_cache.update(df_suggested.index.astype(str).tolist())
    smiles_to_cache.update(df_alternatives.index.astype(str).tolist())
    smiles_to_cache.update(df_neutral.index.astype(str).tolist())
    smiles_to_cache.update(df_cut.index.astype(str).tolist())
    for smiles in smiles_to_cache:
        _get_smiles_image_src(smiles)

    fig = go.Figure()

    if not df_neutral.empty:
        fig.add_trace(
            go.Scattergl(
                x=df_neutral["UMAP1"],
                y=df_neutral["UMAP2"],
                mode="markers",
                name="unseen",
                marker={"size": 8,
                        "color": "#9aa0a6",
                        "opacity": 0.75,
                        "line": {"width": 0.5, "color": "black"}},
                customdata=np.column_stack(
                    [
                        df_neutral.index.astype(str),
                        np.full(len(df_neutral), "PENDING"),
                        np.full(len(df_neutral), "unseen"),
                    ]
                ),
                hovertemplate=(
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not df_cut.empty:
        fig.add_trace(
            go.Scattergl(
                x=df_cut["UMAP1"],
                y=df_cut["UMAP2"],
                mode="markers",
                name="pruned",
                marker={"size": 10, "symbol": "x", "color": "#6f42c1", "opacity": 0.8},
                customdata=np.column_stack(
                    [
                        df_cut.index.astype(str),
                        np.full(len(df_cut), "PENDING"),
                        np.full(len(df_cut), "pruned"),
                    ]
                ),
                hovertemplate=(
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not df_alternatives.empty:
        alt_sizes = pd.to_numeric(df_alternatives.get("priority", 0.6), errors="coerce").fillna(0.6)
        alt_sizes = (15 * alt_sizes).clip(lower=5, upper=15)
        fig.add_trace(
            go.Scattergl(
                x=df_alternatives["UMAP1"],
                y=df_alternatives["UMAP2"],
                mode="markers",
                name="alternatives",
                marker={
                    "size": alt_sizes,
                    "symbol": "diamond",
                    "color": "#188F9D",
                    "opacity": 0.85,
                    "line": {"width": 1.5, "color": "black"},
                },
                customdata=np.column_stack(
                    [
                        df_alternatives.index.astype(str),
                        np.full(len(df_alternatives), "PENDING"),
                        np.full(len(df_alternatives), "alternative"),
                    ]
                ),
                hovertemplate=(
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not df_suggested.empty:
        fig.add_trace(
            go.Scattergl(
                x=df_suggested["UMAP1"],
                y=df_suggested["UMAP2"],
                mode="markers",
                name="suggested",
                marker={
                    "size": 14,
                    "symbol": "square",
                    "color": "#188F9D",
                    "opacity": 0.95,
                    "line": {"width": 1.5, "color": "black"},
                },
                customdata=np.column_stack(
                    [
                        df_suggested.index.astype(str),
                        np.full(len(df_suggested), "PENDING"),
                        np.full(len(df_suggested), "suggested"),
                    ]
                ),
                hovertemplate=(
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not df_seen.empty:
        measured_values = pd.to_numeric(df_seen[selected_objective], errors="coerce")
        fig.add_trace(
            go.Scattergl(
                x=df_seen["UMAP1"],
                y=df_seen["UMAP2"],
                mode="markers",
                name="measured",
                marker={
                    "size": 12,
                    "color": measured_values,
                    "colorscale": "RdBu_r",
                    "showscale": True,
                    "colorbar": {"title": {"text": selected_objective, "side": "right"}},
                    "line": {"width": 1, "color": "black"},
                },
                customdata=np.column_stack(
                    [
                        df_seen.index.astype(str),
                        df_seen[selected_objective].astype(str),
                        np.full(len(df_seen), "measured"),
                    ]
                ),
                hovertemplate=(
                    f"{selected_objective}: "
                    "%{customdata[1]}<br>"
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"UMAP Projection ({selected_objective})",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )

    graph_children = _build_umap_graph_children(fig.to_dict())
    return graph_children, fig.to_dict(), _search_space_signature(search_space)


@callback(
    Output("tooltip-umap", "show"),
    Output("tooltip-umap", "bbox"),
    Output("tooltip-umap", "children"),
    Input("graph-umap", "hoverData"),
    prevent_initial_call=True,
)
def render_umap_tooltip(hover_data):
    """Render RDKit structure tooltip for hovered UMAP points."""
    if not hover_data or not hover_data.get("points"):
        return False, no_update, no_update

    point = hover_data["points"][0]
    bbox = point.get("bbox", {})
    custom_data = point.get("customdata", [])
    if not custom_data:
        return False, no_update, no_update

    smiles = custom_data[0]
    objective_value = custom_data[1] if len(custom_data) > 1 else ""
    status = custom_data[2] if len(custom_data) > 2 else ""

    tooltip_children = html.Div(
        [
            _smiles_viewer(smiles),
            html.Div(f"Status: {status}", style={"marginTop": "0.35rem"}),
            html.Div(f"Value: {objective_value}"),
        ],
        style={"maxWidth": "280px"},
    )

    return True, bbox, tooltip_children


####################################
# callbacks for SHAP analysis
####################################

@callback(
    Output("shap-objective-dropdown", "options"),
    Output("shap-objective-dropdown", "value"),
    Input("store-objectives", "data"),
    State("shap-objective-dropdown", "value"),
)
def populate_shap_objective_dropdown(objectives_dict, current_value):
    """Populate SHAP objective dropdown from selected objectives."""
    return _objective_dropdown_payload(objectives_dict, current_value, action = "analyze")


@callback(
    Output("btn-refresh-shap", "disabled"),
    Input("shap-objective-dropdown", "value"),
)
def toggle_shap_refresh_button(selected_objective):
    """Enable SHAP refresh button only after objective selection."""
    return (
        selected_objective is None
        or str(selected_objective).startswith("Select objective to")
    )


@callback(
    Output("alert-shap-stale", "children"),
    Input("store-search-space", "data"),
    Input("store-shap-searchspace-signature", "data"),
)
def render_shap_stale_warning(search_space, stored_signature):
    """Warn when the saved SHAP plots no longer reflect the current search space."""
    return _render_stale_plot_warning(search_space, stored_signature, "analysis")


@callback(
    Output("visual-shap", "children"),
    Output("store-shap-beeswarm-figure", "data"),
    Output("store-shap-bar-figure", "data"),
    Output("store-shap-searchspace-signature", "data"),
    Input("btn-refresh-shap", "n_clicks"),
    State("store-search-space", "data"),
    State("store-objectives", "data"),
    State("shap-objective-dropdown", "value"),
    prevent_initial_call=True,
)
def shap_visualization(_refresh_clicks, search_space, objectives_dict, selected_objective):
    """Render Plotly SHAP beeswarm and mean absolute SHAP bar plots."""

    if not _refresh_clicks:
        return no_update, no_update, no_update, no_update

    if not search_space or not objectives_dict:
        return dbc.Alert(
            "No search space or objectives available for SHAP analysis.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    objectives = [obj["name"] for obj in objectives_dict if obj.get("name", "").strip()]
    objective_modes = {obj["name"]: obj["mode"] for obj in objectives_dict}
    if not objectives:
        return dbc.Alert(
            "No valid objectives available for SHAP analysis.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    search_space_df = pd.read_json(search_space, orient='split')

    if selected_objective not in search_space_df.columns:
        return dbc.Alert(
            f"Selected objective '{selected_objective}' not found in the search space. Please select a valid objective.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update
    
    # check if the selected objective has any measured values
    if search_space_df[selected_objective].astype(str).eq("PENDING").all():
        return dbc.Alert(
            f"No measured values found for the selected objective '{selected_objective}'. Please report results before running SHAP analysis.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update


    # remove the other objectives apart from the selected one
    for obj in objectives:
        if obj != selected_objective and obj in search_space_df.columns:
            search_space_df = search_space_df.drop(columns=[obj])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # timestamp for unique filename
    temp_filename = f"scopebo_webapp_{timestamp}.csv"
    search_space_df.to_csv(temp_filename, index=True)
    try:
        shap_values,_ = ScopeBO().feature_analysis(
            filename=f"scopebo_webapp_{timestamp}.csv", 
            objectives=[selected_objective],
            objective_mode={selected_objective: objective_modes.get(selected_objective, "max")},
            plot_type=[],
        )
    except Exception as e:
        return _build_traceback_alert("doing SHAP analysis", e), no_update, no_update, no_update
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    shap_array = np.asarray(shap_values.values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]
    if shap_array.ndim != 2:
        return dbc.Alert(
            "Unsupported SHAP output format for plotting.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    n_samples, n_features = shap_array.shape
    feature_names = list(getattr(shap_values, "feature_names", []))
    if len(feature_names) != n_features:
        feature_names = [f"Feature {i + 1}" for i in range(n_features)]

    shap_data = np.asarray(getattr(shap_values, "data", np.full((n_samples, n_features), np.nan)))
    if shap_data.ndim != 2 or shap_data.shape != (n_samples, n_features):
        shap_data = np.full((n_samples, n_features), np.nan)

    # Build an explicit, deterministic sample mapping for SHAP hover tooltips.
    if selected_objective in search_space_df.columns:
        measured_mask = search_space_df[selected_objective].astype(str).ne("PENDING")
        measured_df = search_space_df.loc[measured_mask]
    else:
        measured_df = search_space_df

    measured_smiles = measured_df.index.astype(str).tolist()
    if len(measured_smiles) != n_samples:
        measured_smiles = [f"sample_{i + 1}" for i in range(n_samples)]
    sample_ids = [f"sample_{i + 1}" for i in range(n_samples)]

    for smiles in measured_smiles:
        _get_smiles_image_src(smiles)

    mean_abs = pd.Series(np.nanmean(np.abs(shap_array), axis=0), index=feature_names)
    mean_abs = mean_abs.sort_values(ascending=False)
    top_features = mean_abs.index.tolist()[:10]  # limit to top 10 features for plots
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    rng = np.random.default_rng(42)
    beeswarm_rows = []
    for rank, feature in enumerate(top_features):
        col_idx = feature_to_idx[feature]
        shap_col = shap_array[:, col_idx]
        feat_val_col = shap_data[:, col_idx]
        jitter = rng.uniform(-0.15, 0.15, size=n_samples)
        for i in range(n_samples):
            beeswarm_rows.append(
                {
                    "sample_id": sample_ids[i],
                    "feature": feature,
                    "feature_rank": rank,
                    "y": rank + jitter[i],
                    "shap_value": shap_col[i],
                    "feature_value": feat_val_col[i],
                    "smiles": measured_smiles[i],
                }
            )

    df_beeswarm = pd.DataFrame(beeswarm_rows)
    has_numeric_feature_color = pd.to_numeric(df_beeswarm["feature_value"], errors="coerce").notna().any()

    if has_numeric_feature_color:
        marker_dict = {
            "size": 7,
            "opacity": 0.8,
            "line": {"width": 0.2, "color": "black"},
            "color": pd.to_numeric(df_beeswarm["feature_value"], errors="coerce"),
            "colorscale": "RdBu_r",
            "showscale": True,
            "colorbar": {"title": {"text": "Normalized feature value", "side": "right"}},
        }
    else:
        marker_dict = {
            "size": 7,
            "opacity": 0.8,
            "line": {"width": 0.2, "color": "black"},
            "color": "#1f77b4",
        }

    beeswarm_fig = go.Figure(
        go.Scattergl(
            x=df_beeswarm["shap_value"],
            y=df_beeswarm["y"],
            mode="markers",
            marker=marker_dict,
            customdata=np.column_stack(
                [
                    df_beeswarm["sample_id"].astype(str),
                    df_beeswarm["smiles"].astype(str),
                    df_beeswarm["feature"].astype(str),
                    df_beeswarm["feature_value"].astype(str),
                ]
            ),
            hovertemplate=(
                "Sample: %{customdata[0]}<br>"
                "Feature: %{customdata[2]}<br>"
                "SHAP: %{x:.4f}<br>"
                "Feature value: %{customdata[3]}<extra></extra>"
            ),
            name="",
        )
    )
    beeswarm_fig.update_layout(
        title=f"SHAP Beeswarm (Top 10 Features) - {selected_objective}",
        xaxis_title="SHAP value",
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(len(top_features))),
            "ticktext": top_features,
            "autorange": "reversed",
        },
        template="plotly_white",
        height=700,
        margin={"l": 220, "r": 20, "t": 60, "b": 40},
        showlegend=False,
    )
    beeswarm_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    bar_features = mean_abs.index[:10][::-1]
    bar_values = mean_abs.values[:10][::-1]
    bar_fig = go.Figure(
        go.Bar(
            x=bar_values,
            y=bar_features,
            orientation="h",
            marker={"color": "#1561C2"},
            hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    bar_fig.update_layout(
        title=f"Mean Absolute SHAP Values (Top 10) - {selected_objective}",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=650,
        margin={"l": 220, "r": 20, "t": 60, "b": 40},
    )

    beeswarm_fig_dict = beeswarm_fig.to_dict()
    bar_fig_dict = bar_fig.to_dict()
    return (
        _build_shap_graph_children(beeswarm_fig_dict, bar_fig_dict),
        beeswarm_fig_dict,
        bar_fig_dict,
        _search_space_signature(search_space),
    )


@callback(
    Output("tooltip-shap-beeswarm", "show"),
    Output("tooltip-shap-beeswarm", "bbox"),
    Output("tooltip-shap-beeswarm", "children"),
    Input("graph-shap-beeswarm", "hoverData"),
    prevent_initial_call=True,
)
def render_shap_tooltip(hover_data):
    """Render RDKit structure tooltip for hovered SHAP beeswarm points."""
    if not hover_data or not hover_data.get("points"):
        return False, no_update, no_update

    point = hover_data["points"][0]
    bbox = point.get("bbox", {})
    custom_data = point.get("customdata", [])
    if not custom_data:
        return False, no_update, no_update

    sample_id = custom_data[0] if len(custom_data) > 0 else ""
    smiles = custom_data[1] if len(custom_data) > 1 else ""
    feature_name = custom_data[2] if len(custom_data) > 2 else ""
    feature_value = custom_data[3] if len(custom_data) > 3 else ""
    shap_value = point.get("x", "")

    tooltip_children = html.Div(
        [
            _smiles_viewer(smiles),
            html.Div(f"Sample: {sample_id}", style={"marginTop": "0.35rem"}),
            html.Div(f"Feature: {feature_name}", style={"marginTop": "0.35rem"}),
            html.Div(f"Feature value: {feature_value}"),
            html.Div(f"SHAP: {shap_value}"),
        ],
        style={"maxWidth": "280px"},
    )

    return True, bbox, tooltip_children


####################################
# callbacks for predictive modeling
####################################

@callback(
    Output("pred-objective-dropdown", "options"),
    Output("pred-objective-dropdown", "value"),
    Input("store-objectives", "data"),
    State("pred-objective-dropdown", "value"),
)
def populate_pred_objective_dropdown(objectives_dict, current_value):
    """Populate predictive-modeling objective dropdown from selected objectives."""
    return _objective_dropdown_payload(objectives_dict, current_value, action = "predict")


@callback(
    Output("btn-refresh-pred", "disabled"),
    Input("pred-objective-dropdown", "value"),
)
def toggle_pred_refresh_button(selected_objective):
    """Enable prediction refresh button only after objective selection."""
    return (
        selected_objective is None
        or str(selected_objective).startswith("Select objective to")
    )


@callback(
    Output("alert-pred-stale", "children"),
    Input("store-search-space", "data"),
    Input("store-pred-searchspace-signature", "data"),
)
def render_prediction_stale_warning(search_space, stored_signature):
    """Warn when the saved prediction plot no longer reflects the current search space."""
    return _render_stale_plot_warning(search_space, stored_signature, "prediction plot")


@callback(
    Output("visual-pred", "children"),
    Output("store-pred-results", "data"),
    Output("store-pred-figure", "data"),
    Output("store-pred-searchspace-signature", "data"),
    Input("btn-refresh-pred", "n_clicks"),
    State("store-search-space", "data"),
    State("store-objectives", "data"),
    State("pred-objective-dropdown", "value"),
    prevent_initial_call=True,
)
def pred_visualization(_refresh_clicks, search_space, objectives_dict, selected_objective):
    """Create predictions and visualize them."""

    if not _refresh_clicks:
        return no_update, no_update, no_update, no_update

    if not search_space or not objectives_dict:
        return dbc.Alert(
            "No search space or objectives available for prediction visualization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    objectives = [obj["name"] for obj in objectives_dict if obj.get("name", "").strip()]
    objective_modes = {obj["name"]: obj["mode"] for obj in objectives_dict}
    if not objectives:
        return dbc.Alert(
            "No valid objectives available for prediction visualization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    search_space_df = pd.read_json(search_space, orient='split')
    # remove the other objectives apart from the selected one
    for obj in objectives:
        if obj != selected_objective and obj in search_space_df.columns:
            search_space_df = search_space_df.drop(columns=[obj])

    if selected_objective not in search_space_df.columns:
        return dbc.Alert(
            "Could not find the selected objective in the current search space. Please select a valid objective.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    # return alert if no experimental results
    if search_space_df[selected_objective].astype(str).eq("PENDING").all():
        return dbc.Alert(
            "No experimental results available for the selected objective. Please add some measured data before generating predictions.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # timestamp for unique filename
    temp_filename = f"scopebo_webapp_{timestamp}.csv"
    search_space_df.to_csv(temp_filename, index=True)
    df_pred = None

    # generate predictions using ScopeBO's expected_improvement method
    try:
        df_pred = ScopeBO().expected_improvement(
            filename=f"scopebo_webapp_{timestamp}.csv",
            objectives = [selected_objective],
            objective_mode={selected_objective: objective_modes.get(selected_objective, "max")},
            results_filename = None,
            visualize = False
            )
    except Exception as e:
        return _build_traceback_alert("generating predictions", e), no_update, no_update, no_update
    
    # generate UMAP coords for the visualization of the predictions
    try:
        df_dict = ScopeBO().visualize(
            filename=temp_filename,
            obj_to_show=selected_objective,
            objectives=[selected_objective],
            draw_structures=False,
            show_figure=False,
            return_dfs=True,
        )
    except Exception as e:  # provide feedback on the error if ScopeBO.visualize() fails
        return _build_traceback_alert("building UMAP", e), no_update, no_update, no_update
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    df_seen = df_dict.get("seen", pd.DataFrame()).copy()
    
    # combine the data for all other samples
    df_unseen = pd.concat(
        [
            df_dict.get("suggested", pd.DataFrame()).copy(),
            df_dict.get("alternatives", pd.DataFrame()).copy(),
            df_dict.get("neutral", pd.DataFrame()).copy(),
            df_dict.get("cut", pd.DataFrame()).copy(),
        ],
        axis=0,
    )

    pred_col = f"Prediction_{selected_objective}"
    std_col = f"Std. dev. of pred._{selected_objective}"
    if pred_col not in df_pred.columns or std_col not in df_pred.columns:
        return dbc.Alert(
            "Prediction output is missing required columns for visualization.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), no_update, no_update, no_update

    # Align prediction outputs by SMILES index and attach to unseen dataframe.
    df_unseen[selected_objective] = pd.to_numeric(df_pred.reindex(df_unseen.index)[pred_col], errors="coerce")
    df_unseen[f"StdDev_{selected_objective}"] = pd.to_numeric(df_pred.reindex(df_unseen.index)[std_col], errors="coerce")

    # Marker size encodes uncertainty: larger marker = lower uncertainty.
    size_min, size_max = 3.5, 10
    std_vals = df_unseen[f"StdDev_{selected_objective}"].to_numpy(dtype=float)
    finite_mask = np.isfinite(std_vals)
    if finite_mask.any():
        std_min = np.nanmin(std_vals[finite_mask])
        std_max = np.nanmax(std_vals[finite_mask])
        if std_max - std_min == 0:
            size_vals = np.full(len(df_unseen), (size_min + size_max) / 2)
        else:
            size_vals = size_min + (std_max - std_vals) * (size_max - size_min) / (std_max - std_min)
            size_vals = np.where(np.isfinite(size_vals), size_vals, (size_min + size_max) / 2)
    else:
        size_vals = np.full(len(df_unseen), (size_min + size_max) / 2)
    df_unseen["size"] = size_vals

    # Keep color scaling consistent across predicted and measured points.
    predicted_values = pd.to_numeric(df_unseen[selected_objective], errors="coerce")
    measured_values = pd.to_numeric(df_seen[selected_objective], errors="coerce") if selected_objective in df_seen.columns else pd.Series(dtype=float)
    all_values = pd.concat([predicted_values, measured_values], ignore_index=True)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.empty:
        cmin, cmax = 0.0, 1.0
    else:
        cmin, cmax = float(finite_values.min()), float(finite_values.max())
        if cmin == cmax:
            cmin -= 1.0
            cmax += 1.0

    for smiles in df_unseen.index.astype(str):
        _get_smiles_image_src(smiles)
    for smiles in df_seen.index.astype(str):
        _get_smiles_image_src(smiles)

    fig = go.Figure()

    if not df_unseen.empty:
        fig.add_trace(
            go.Scattergl(
                x=df_unseen["UMAP1"],
                y=df_unseen["UMAP2"],
                mode="markers",
                name="predicted",
                marker={
                    "size": df_unseen["size"],
                    "color": predicted_values,
                    "colorscale": "RdBu_r",
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": True,
                    "colorbar": {"title": {"text": selected_objective, "side": "right"}},
                    "line": {"width": 1, "color": "black"},
                    "opacity": 1.0,
                },
                customdata=np.column_stack(
                    [
                        df_unseen.index.astype(str),
                        predicted_values.astype(str),
                        df_unseen[f"StdDev_{selected_objective}"].astype(str),
                        np.full(len(df_unseen), "predicted"),
                    ]
                ),
                hovertemplate=(
                    f"{selected_objective}: "
                    "%{customdata[1]}<br>"
                    "StdDev: %{customdata[2]}<br>"
                    "Status: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    if not df_seen.empty and selected_objective in df_seen.columns:
        fig.add_trace(
            go.Scattergl(
                x=df_seen["UMAP1"],
                y=df_seen["UMAP2"],
                mode="markers",
                name="measured",
                marker={
                    "size": 10,
                    "symbol": "square",
                    "color": measured_values,
                    "colorscale": "RdBu_r",
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": False,
                    "line": {"width": 1.8, "color": "black"},
                    "opacity": 1.0,
                },
                customdata=np.column_stack(
                    [
                        df_seen.index.astype(str),
                        df_seen[selected_objective].astype(str),
                        np.full(len(df_seen), "measured"),
                    ]
                ),
                hovertemplate=(
                    f"{selected_objective}: "
                    "%{customdata[1]}<br>"
                    "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Predicted UMAP Projection ({selected_objective})",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )

    pred_fig_dict = fig.to_dict()
    return (
        _build_pred_graph_children(pred_fig_dict),
        df_pred.to_json(date_format="iso", orient="split"),
        pred_fig_dict,
        _search_space_signature(search_space),
    )


@callback(
    Output("tooltip-pred", "show"),
    Output("tooltip-pred", "bbox"),
    Output("tooltip-pred", "children"),
    Input("graph-pred", "hoverData"),
    prevent_initial_call=True,
)
def render_pred_tooltip(hover_data):
    """Render RDKit structure tooltip for hovered prediction UMAP points."""
    if not hover_data or not hover_data.get("points"):
        return False, no_update, no_update

    point = hover_data["points"][0]
    bbox = point.get("bbox", {})
    custom_data = point.get("customdata", [])
    if not custom_data:
        return False, no_update, no_update

    smiles = custom_data[0]
    value = custom_data[1] if len(custom_data) > 1 else ""
    status = custom_data[3] if len(custom_data) > 3 else (custom_data[2] if len(custom_data) > 2 else "")
    std_dev = custom_data[2] if len(custom_data) > 3 else ""

    details = [
        _smiles_viewer(smiles),
        html.Div(f"Status: {status}", style={"marginTop": "0.35rem"}),
        html.Div(f"Value: {value}"),
    ]
    if std_dev:
        details.append(html.Div(f"StdDev: {std_dev}"))

    tooltip_children = html.Div(details, style={"maxWidth": "280px"})
    return True, bbox, tooltip_children


@callback(
    Output("btn-download-pred", "disabled"),
    Output("card-pred-download", "style"),
    Input("store-pred-results", "data"),
)
def toggle_prediction_download(pred_results):
    """Show prediction download card and enable button once predictions are available."""
    if pred_results:
        return False, {**OUTER_CARD_STYLE, "display": "block"}
    return True, {"display": "none"}


@callback(
    Output("download-predictions", "data"),
    Output("feedback-download-pred", "children"),
    Input("btn-download-pred", "n_clicks"),
    State("store-pred-results", "data"),
    State("input-dwl-pred-filename", "value"),
    prevent_initial_call=True,
)
def download_predictions(n_clicks, pred_results, filename):
    """Download the latest prediction dataframe as a CSV file."""
    if not n_clicks:
        return no_update, no_update

    if not pred_results:
        return no_update, dbc.Alert(
            "No predictions available to download yet.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    df_pred = pd.read_json(pred_results, orient="split")

    raw_name = (filename or "").strip()
    if raw_name:
        safe_name = os.path.basename(raw_name).replace("\\", "_").replace("/", "_")
    else:
        safe_name = f"scopebo_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    return dcc.send_data_frame(df_pred.to_csv, safe_name, index=True), dbc.Alert(
        f"Download successful: {safe_name}",
        style=SUCCESS_ALERT_STYLE,
        dismissable=False,
    )


####################################
# callbacks for featurization
####################################

@callback(
    Output('page-featurization', 'style', allow_duplicate=True),
    Output('page-upload', 'style', allow_duplicate=True),
    Input('btn-back-to-upload-from-featurization', 'n_clicks'),
    prevent_initial_call=True,
)
def go_back_to_upload_from_featurization(n_clicks):
    """Navigate from featurization page back to upload page (starting page)."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


@callback(
    Output("feedback-upload-smiles", "children"),
    Output("store-smiles-upload-valid", "data"),
    Input("upload-smiles", "contents"),
    State("upload-smiles", "filename"),
    prevent_initial_call=True,
)
def validate_smiles_upload(contents, filename):
    """Provide upload feedback for the SMILES list used in featurization."""
    if contents is None:
        return no_update, False

    if not filename:
        return dbc.Alert(
            "Upload failed: missing filename.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), False

    if not filename.lower().endswith(".csv"):
        return dbc.Alert(
            f"Invalid file type: {filename}. Please upload a CSV file.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), False

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df_smiles = pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0)
        df_smiles = _drop_fully_empty_rows(df_smiles)
    except Exception:
        return dbc.Alert(
            "Could not read the uploaded file. Please provide a valid CSV.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), False

    if df_smiles.empty:
        return dbc.Alert(
            "The uploaded CSV is empty.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), False
    
    smiles_columns = [col for col in df_smiles.columns if str(col).strip().lower() == "smiles"]
    if not smiles_columns:
        return dbc.Alert(
            "The uploaded CSV must contain a column named 'smiles'.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        ), False

    return dbc.Alert(
        f"Successfully uploaded SMILES file: {filename} ({len(df_smiles)} rows).",
        style=SUCCESS_ALERT_STYLE,
        dismissable=False,
    ), True


@callback(
    Output("btn-featurization", "disabled"),
    Input("store-smiles-upload-valid", "data"),
    Input("interval-featurization-progress", "n_intervals"),
    State("store-featurization-trigger", "data"),
)
def toggle_featurization_button(upload_valid, _n_intervals, trigger_data):
    """Enable featurization button only after successful SMILES upload validation."""
    run_id = (trigger_data or {}).get("run_id")
    is_running = _is_featurization_job_running(run_id)
    return (not bool(upload_valid)) or is_running


@callback(
    Output("common-core-preview", "children"),
    Output("store-featurization-trigger", "data"),
    Output("feedback-featurization-run", "children", allow_duplicate=True),
    Input("btn-featurization", "n_clicks"),
    State("upload-smiles", "contents"),
    State("upload-smiles", "filename"),
    prevent_initial_call=True,
)
def show_common_core_and_trigger_featurization(n_clicks, upload_contents, upload_filename):
    """Show common core preview first, then trigger long-running featurization callback."""
    if not n_clicks:
        return no_update, no_update, no_update

    smiles_list, parse_error = _extract_smiles_from_upload(upload_contents, upload_filename)
    if parse_error is not None:
        return no_update, no_update, parse_error

    try:
        _, template_smarts, preview_src = build_common_core_preview(smiles_list)
    except Exception as e:
        return no_update, no_update, _build_traceback_alert("building common-core preview", e)

    preview = html.Div(
        [
            html.Div("Common core used for atom-level descriptors:", style={"fontWeight": "bold"}),
            html.Div(f"SMARTS: {template_smarts}", style={"fontFamily": "monospace", "fontSize": "0.9rem"}),
            html.Img(src=preview_src, width=320, height=180, style={"marginTop": "0.5rem", "border": "1px solid #dee2e6"}),
        ],
        style={"maxWidth": "540px", "margin": "0.5rem auto 0 auto", "textAlign": "center"},
    )

    trigger_payload = {
        "run_id": datetime.now().isoformat(),
        "smiles_list": smiles_list,
    }
    return preview, trigger_payload, dbc.Alert(
        "Common core identified. Starting featurization...",
        style=INFO_ALERT_STYLE,
        dismissable=False,
    )


@callback(
    Output("feedback-featurization-run", "children", allow_duplicate=True),
    Input("store-featurization-trigger", "data"),
    prevent_initial_call=True,
)
def run_smiles_featurization(trigger_data):
    """Start a background featurization job so progress can update live."""
    if not trigger_data:
        return no_update

    run_id = trigger_data.get("run_id")
    smiles_list = trigger_data.get("smiles_list", [])
    if not run_id:
        return dbc.Alert(
            "Missing featurization run ID. Please retry.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )
    if not smiles_list:
        return dbc.Alert(
            "No valid SMILES strings available for featurization.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    if _has_active_featurization_job():
        return dbc.Alert(
            "A featurization job is already running. Please wait until it finishes before starting a new one.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    thread = threading.Thread(
        target=_run_featurization_job,
        args=(run_id, smiles_list),
        daemon=True,
    )
    thread.start()

    return dbc.Alert(
        "Featurization is running. Track live progress below.",
        style=INFO_ALERT_STYLE,
        dismissable=False,
    )


@callback(
    Output("feedback-featurization-run", "children", allow_duplicate=True),
    Input("btn-interrupt-featurization", "n_clicks"),
    State("store-featurization-trigger", "data"),
    prevent_initial_call=True,
)
def interrupt_featurization_run(n_clicks, trigger_data):
    """Request cancellation for the active featurization run."""
    if not n_clicks:
        return no_update

    run_id = (trigger_data or {}).get("run_id")
    if not run_id:
        return dbc.Alert(
            "No active featurization run found.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    state = _get_featurization_job_state(run_id)
    if state.get("status") != "running":
        return dbc.Alert(
            "No running featurization job to interrupt.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    _set_featurization_job_state(
        run_id,
        cancel_requested=True,
        message="Interrupt requested. Stopping featurization...",
    )
    return dbc.Alert(
        "Interrupt requested. Featurization will stop shortly.",
        style=WARNING_ALERT_STYLE,
        dismissable=False,
    )


@callback(
    Output("interval-featurization-progress", "disabled"),
    Input("store-featurization-trigger", "data"),
    Input("interval-featurization-progress", "n_intervals"),
)
def toggle_featurization_interval(trigger_data, _n_intervals):
    """Run progress polling only while a featurization job is active/pending."""
    run_id = (trigger_data or {}).get("run_id")
    if not run_id:
        return True

    state = _get_featurization_job_state(run_id)
    status = state.get("status")

    if status in {"done", "cancelled", "error"}:
        return True

    # Keep polling while job is queued/running or before first state write lands.
    return False


@callback(
    Output("progress-featurization", "value"),
    Output("progress-featurization", "label"),
    Output("progress-featurization", "animated"),
    Output("progress-featurization", "style"),
    Output("btn-interrupt-featurization", "style"),
    Output("progress-featurization-text", "children"),
    Output("featurization-running-note", "children"),
    Output("store-featurization-data", "data", allow_duplicate=True),
    Output("feedback-featurization-run", "children", allow_duplicate=True),
    Input("interval-featurization-progress", "n_intervals"),
    State("store-featurization-trigger", "data"),
    State("store-featurization-data", "data"),
    prevent_initial_call=True,
)
def update_featurization_progress(_, trigger_data, existing_data):
    """Poll background job state and update progress bar while featurization runs."""
    hidden_style = {"height": "22px", "maxWidth": "540px", "margin": "0.5rem auto", "display": "none"}
    visible_style = {"height": "22px", "maxWidth": "540px", "margin": "0.5rem auto", "display": "block"}
    hidden_interrupt_style = {"display": "none"}
    visible_interrupt_style = {"display": "inline-block", "backgroundColor": color_bad, "borderColor": color_bad,}
    running_note = dbc.Alert(
        "The featurization process may take some hours, but you can interrupt it and resume later (results are saved after every 5 SMILES strings).",
        style=WARNING_ALERT_STYLE,
        dismissable=False,
        className="text-center",
    )

    if not trigger_data or "run_id" not in trigger_data:
        return 0, "0%", False, hidden_style, hidden_interrupt_style, "", None, no_update, no_update

    run_id = trigger_data["run_id"]
    state = _get_featurization_job_state(run_id)
    if not state:
        return 0, "0%", True, visible_style, visible_interrupt_style, "Preparing featurization job...", running_note, no_update, no_update

    status = state.get("status", "running")
    percent = int(state.get("percent", 0))
    done = int(state.get("done", 0))
    total = int(state.get("total", 0))
    message = state.get("message", "")

    # Emit terminal states only once to avoid repeated re-rendering every interval tick.
    if status in {"done", "cancelled", "error"} and state.get("result_emitted"):
        return no_update, no_update, no_update, no_update, no_update, no_update, None, no_update, no_update

    if status == "done":
        row_count = state.get("row_count", 0)
        descriptors_json = state.get("descriptors_json")
        updated_store_data = _append_featurization_store(existing_data, descriptors_json, "ScopeBO")
        _set_featurization_job_state(run_id, result_emitted=True)
        done_text = message or f"Featurization completed successfully for {row_count} substrate(s)."
        return (
            100,
            "100%",
            False,
            visible_style,
            hidden_interrupt_style,
            done_text,
            None,
            updated_store_data,
            dbc.Alert(
                [
                    f"Featurization completed successfully for {row_count} substrate(s). You can download the descriptors below.",
                    html.Br(),
                    "If you need to featurize another reactant list, please upload a new SMILES CSV and click 'Featurize SMILES' again.",
                ],
                style=SUCCESS_ALERT_STYLE,
                dismissable=False,
            ),
        )

    if status == "cancelled":
        cancelled_text = state.get("message", "Featurization interrupted by user.")
        _set_featurization_job_state(run_id, result_emitted=True)
        return (
            percent,
            f"{percent}%",
            False,
            visible_style,
            hidden_interrupt_style,
            cancelled_text,
            None,
            no_update,
            dbc.Alert(
                cancelled_text,
                style=WARNING_ALERT_STYLE,
                dismissable=False,
            ),
        )

    if status == "error":
        error_text = state.get("message", "Featurization failed.")
        _set_featurization_job_state(run_id, result_emitted=True)
        return (
            0,
            "0%",
            False,
            visible_style,
            hidden_interrupt_style,
            error_text,
            None,
            no_update,
            state.get("error", dbc.Alert("Featurization failed.", style=DANGER_ALERT_STYLE, dismissable=False)),
        )

    label = f"{percent}%"
    progress_text = message or (f"Processed {done}/{total}" if total else "Running featurization...")
    return percent, label, True, visible_style, visible_interrupt_style, progress_text, running_note, no_update, no_update


@callback(
    Output("card-featurization-download", "style"),
    Output("dropdown-featurization-dataset", "options"),
    Output("dropdown-featurization-dataset", "value"),
    Output("container-go-to-searchspace-from-featurization", "style"),
    Input("store-featurization-data", "data"),
)
def populate_featurization_download_controls(featurization_data):
    """Show download card and populate the selector once one or more
    featurization outputs exist."""

    datasets = featurization_data or []

    if not datasets:
        return (
            {"display": "none"},
            [],
            None,
            {"display": "none", "textAlign": "center"},
        )

    options = [
        {
            "label": "Select featurization result",
            "value": None,
        }
    ]

    for idx, dataset_json in enumerate(datasets):
        dataset_data = dataset_json.get("data") if isinstance(dataset_json, dict) else None
        dataset_source = dataset_json.get("source", "Unknown") if isinstance(dataset_json, dict) else "Unknown"
        row_count_text = ""
        try:
            row_count = len(pd.read_json(dataset_data, orient="split"))
            row_count_text = f"({row_count} rows)"
        except Exception:
            pass

        options.append(
            {
                "label": f"Dataset {idx + 1} - {row_count_text}",
                "value": idx,
            }
        )

    return (
        {**OUTER_CARD_STYLE, "display": "block"},
        options,
        None,
        {"display": "block", "textAlign": "center"},
    )


@callback(
    Output("btn-download-featurization", "disabled"),
    Input("dropdown-featurization-dataset", "value"),
)
def enable_download_featurization(selected_value):
    return selected_value is None


@callback(
    Output('page-featurization', 'style', allow_duplicate=True),
    Output('page-preprocess', 'style', allow_duplicate=True),
    Input('btn-go-to-searchspace-from-featurization', 'n_clicks'),
    prevent_initial_call=True,
)
def go_to_search_space_creation(n_clicks):
    """Navigate from featurization page to preprocess page."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


@callback(
    Output("download-featurization", "data"),
    Output("feedback-download-featurization", "children"),
    Input("btn-download-featurization", "n_clicks"),
    State("store-featurization-data", "data"),
    State("dropdown-featurization-dataset", "value"),
    State("input-dwl-featurization-filename", "value"),
    prevent_initial_call=True,
)
def download_featurization_data(n_clicks, featurization_data, selected_dataset_idx, filename):
    """Download one selected featurization dataframe from the in-memory store."""
    if not n_clicks:
        return no_update, no_update

    datasets = featurization_data or []

    if not datasets:
        return no_update, dbc.Alert(
            "No featurization data available to download yet.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    selected_idx = int(selected_dataset_idx) if selected_dataset_idx is not None else 0
    if selected_idx < 0 or selected_idx >= len(datasets):
        return no_update, dbc.Alert(
            "Please select a valid featurization result to download.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    selected_dataset = datasets[selected_idx]
    selected_dataset_json = selected_dataset.get("data") if isinstance(selected_dataset, dict) else None
    selected_source = selected_dataset.get("source", "Dataset") if isinstance(selected_dataset, dict) else "Dataset"

    try:
        df_featurization = pd.read_json(selected_dataset_json, orient="split")
    except Exception:
        return no_update, dbc.Alert(
            "Could not parse the selected featurization dataset for download.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    raw_name = (filename or "").strip()
    if raw_name:
        safe_name = os.path.basename(raw_name).replace("\\", "_").replace("/", "_")
    else:
        source_slug = "scopebo" if selected_source == "ScopeBO" else "upload"
        safe_name = f"scopebo_{source_slug}_{selected_idx + 1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    return dcc.send_data_frame(df_featurization.to_csv, safe_name, index=True), dbc.Alert(
        f"Download successful: {safe_name}",
        style=SUCCESS_ALERT_STYLE,
        dismissable=False,
    )


####################################
# callbacks for preprocessing
####################################

@callback(
    Output('page-preprocess', 'style', allow_duplicate=True),
    Output('page-featurization', 'style', allow_duplicate=True),
    Input('btn-back-to-featurization-from-preprocess', 'n_clicks'),
    prevent_initial_call=True,
)
def go_back_to_featurization_from_preprocess(n_clicks):
    """Navigate from preprocess page back to featurization page."""
    if n_clicks:
        return {'display': 'none'}, {'display': 'block'}
    return no_update, no_update


@callback(
    Output("store-featurization-data", "data", allow_duplicate=True),
    Output("feedback-preprocess-upload", "children"),
    Input("upload-preprocess-features", "contents"),
    State("upload-preprocess-features", "filename"),
    State("store-featurization-data", "data"),
    prevent_initial_call=True,
)
def add_uploaded_features(contents, filenames, store):
    """
    Upload and check featurization files on the preprocess page.
    Add them to the featurization data store if valid.
    """

    store = _normalize_store(store or [])

    if contents is None:
        return no_update, no_update

    if store is None:
        store = []

    if not isinstance(contents, list):
        contents = [contents]
        filenames = [filenames]

    uploaded = []

    for content, filename in zip(contents, filenames):

        # Check 1: CSV file
        if not filename.lower().endswith(".csv"):
            return (
                no_update,
                dbc.Alert(
                    f"{filename}: Please upload the data as a CSV file.",
                    color="danger",
                    dismissable=True,
                ),
            )
        
        _, content_string = content.split(",")
        decoded = base64.b64decode(content_string)

        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), index_col=0)
        except Exception as e:
            return _build_traceback_alert(f"reading file {filename}", e)

        # Remove empty rows/columns
        df = _drop_fully_empty_rows(df)

        # Check 2: Index contains valid SMILES
        invalid = [
            smi
            for smi in df.index.astype(str)
            if Chem.MolFromSmiles(smi) is None
        ]
        if invalid:
            return (
                no_update,
                dbc.Alert(
                    f"{filename}: Index contains invalid SMILES "
                    f"(first invalid: '{invalid[0]}').",
                    style=DANGER_ALERT_STYLE,
                    dismissable=False,
                ),
            )

        uploaded.append(filename)

        store.append(
            {
                "data": df.to_json(orient="split"),
                "source": "Upload",
                "filename": filename.split(".")[:-1][0],  # remove file extension
            }
        )

    return (
        store,
        dbc.Alert(
            f"Successfully uploaded {len(uploaded)} file(s): "
            + ", ".join(uploaded),
            style=SUCCESS_ALERT_STYLE,
            dismissable=False,
        ),
    )


@callback(
    Output("feature-summary", "children"),
    Output("card-preprocess-download", "style"),
    Input("store-featurization-data", "data"),
)
def update_feature_summary(store):
    """Display a summary of the uploaded featurization datasets."""

    if not store:
        return dbc.Alert(
            "No feature dataset available. Please upload at least one or featurize on the previous page.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        ), {"display": "none"}

    rows = []
    for i, item in enumerate(store):
        df = pd.read_json(item["data"], orient="split")

        rows.append(
            {
                "id": i,
                "length": len(df),
                "columns": len(df.columns),
                "source": item["source"],
                "display_name": item.get("display_name", f"Dataset {i + 1}"),
                "include": True,
            }
        )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Number of substrates"),
                        html.Th("Number of features"),
                        html.Th("Source"),
                        html.Th("Name (edit if desired)"),
                        html.Th("Include in Search Space?"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(r["length"]),
                            html.Td(r["columns"]),
                            html.Td(r["source"]),
                            html.Td(
                                dcc.Input(
                                    id={"type": "feature-name", "index": r["id"]},
                                    value=r["display_name"],
                                    type="text",
                                    style={"width": "100%", "textAlign": "center"},
                                )
                            ),
                            html.Td(
                                html.Div(
                                    dbc.Checkbox(
                                        id={"type": "feature-include", "index": r["id"]},
                                        value=True,
                                    ),
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                    },
                                )
                            )
                        ]
                    )
                    for r in rows
                ]
            ),
        ],
        className="table",
        style={
        "textAlign": "center",
        "verticalAlign": "middle",
        },
    ), {**OUTER_CARD_STYLE, "display": "block"}


@callback(
    Output("store-featurization-metadata", "data", allow_duplicate=True),
    Input({"type": "feature-name", "index": ALL}, "value"),
    Input({"type": "feature-include", "index": ALL}, "value"),
    State("store-featurization-data", "data"),
    prevent_initial_call=True,
)
def update_feature_metadata(names, includes, store):
    """Make a version of the store with the edited names and include flags for each dataset."""
    
    if not store or not names or not includes:
        return no_update
    
    if len(store) != len(names) or len(store) != len(includes):
        return no_update  # Mismatch in lengths; do not update
    
    for item, name, include in zip(store, names, includes):
        item["display_name"] = (name or "").strip()  # Ensure name is a string and strip whitespace
        item["include"] = bool(include)

    return store


@callback(
    Output("download-preprocess", "data"),
    Output("feedback-download-preprocess", "children"),
    Output("feedback-preprocess-info", "children"),
    Output("table-preprocess-view", "children"),
    Input("btn-create-search-space", "n_clicks"),
    State("input-dwl-preprocess-filename", "value"),
    State("store-featurization-metadata", "data"),
    prevent_initial_call=True,
)
def create_search_space(
    n_clicks,
    output_filename,
    store,
):
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    if not store:
        return (
            no_update,
            dbc.Alert("No feature datasets available.", style=DANGER_ALERT_STYLE),
            no_update,
            no_update,
        )

    # Collect selected feature dataframes
    feature_dfs = {}

    for item in store:

        # Only include datasets that are marked for inclusion
        if not item.get("include", True):
            continue

        name = item["display_name"].strip()
        df = pd.read_json(item["data"], orient="split")
        feature_dfs[name] = df

    if not feature_dfs:
        return (
            no_update,
            dbc.Alert(
                "No feature datasets were selected.",
                style=DANGER_ALERT_STYLE,
            ),
            no_update,
            no_update,
        )
    
    # Validate feature names
    feature_names = list(feature_dfs.keys())

    # Empty names
    empty_names = [name for name in feature_names if not name.strip()]

    if empty_names:
        return (
            no_update,
            dbc.Alert(
                "One or more selected feature sets have an empty name. "
                "Please provide a unique name for every selected feature set.",
                style=DANGER_ALERT_STYLE,
                dismissable=True,
            ),
            no_update,
            no_update,
        )

    # Duplicate names
    duplicates = {
        name
        for name in feature_names
        if feature_names.count(name) > 1
    }

    if duplicates:
        return (
            no_update,
            dbc.Alert(
                "Feature set names must be unique. "
                f"Duplicate name(s): {', '.join(sorted(duplicates))}. Please provide unique names.",
                style=DANGER_ALERT_STYLE,
                dismissable=True,
            ),
            no_update,
            no_update,
        )

    # Create the search space
    try:
        search_space = create_search_space_web(
            reactants = list(feature_dfs.values()),
            reactants_names = feature_names
        )
    except Exception as e:
        return no_update, _build_traceback_alert("creating search space", e), no_update, no_update


    # Filename
    if not output_filename:
        output_filename = f"scopebo_search_space_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    if not output_filename.endswith(".csv"):
        output_filename += ".csv"

    # Download
    download = dcc.send_data_frame(
        search_space.to_csv,
        output_filename,
        index=True,
    )

    # Success alert
    feedback_download = dbc.Alert(
        f"Successfully created and downloaded'{output_filename}'.",
        style=SUCCESS_ALERT_STYLE,
        dismissable=False,
    )

    # Information alert
    feedback_info = dbc.Alert(
        [
            f"The search space contains {len(search_space):,} compounds, ",
            f"{search_space.shape[1]:,} feature columns, ",
            f"generated from {len(feature_dfs)} feature dataset(s).",
        ],
        style=INFO_ALERT_STYLE,
        dismissable=False,
    )

    # Preview table
    preview = dash_table.DataTable(
        data=search_space.reset_index().to_dict("records"),
        columns=[
            {"name": c, "id": c}
            for c in search_space.reset_index().columns
        ],

        page_size=20,
        style_table={
            "overflowX": "auto",
            "borderRadius": "10px",
            "boxShadow": "0px 2px 10px rgba(0,0,0,0.08)",
            "border": "1px solid #e5e7eb",
        },

        style_header={
            "backgroundColor": border_color,
            "color": "white",
            "fontWeight": "600",
            "padding": "10px",
            "border": "none",
            "textAlign": "center",
        },

        style_cell={
            "textAlign": "center",
            "minWidth": "120px",
            "maxWidth": "300px",
            "padding": "10px",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "13px",
            "whiteSpace": "normal",
            "height": "auto",
        },

        style_data={
            "backgroundColor": "white",
            "color": "#111827",
        },

        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f9fafb",
            },
            {
                "if": {"state": "active"},
                "backgroundColor": "#e0f2fe",
                "border": "1px solid #38bdf8",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "#bae6fd",
                "border": "1px solid #0284c7",
            },
        ],
    )

    return (
        download,
        feedback_download,
        feedback_info,
        preview,
    )


@callback(
    Output("btn-create-searchspace", "disabled"),
    Input("store-featurization-data", "data"),
)
def update_create_searchspace_button_disabled(featurization_data):
    """Disable the button unless there is at least one dataframe in the store."""
    datasets = featurization_data or []
    return len(datasets) == 0


####################################
# Helper functions for app callbacks
####################################

# General helpers

def _smiles_viewer(smiles):
    """Generates a visual representation of a molecule from its SMILES string."""
    src = _get_smiles_image_src(smiles)
    if src is None:
        return html.Div(
            "Invalid SMILES",
            style={
                "width": "144px",
                "height": "108px",
                "border": "1px solid",
                "borderColor": color_bad,
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "textAlign": "center",
                "backgroundColor": danger_color,
                "color": color_bad,
                "fontWeight": "bold",
            },
        )

    return html.Img(
        src=src,
        width=144,
        height=108,
    )


def _get_smiles_image_src(smiles, size=(200, 150)):
    """Return a cached data URI for a SMILES structure image, generating on cache miss."""
    smiles = (smiles or "").strip()
    if not smiles:
        return None

    cache_key = (smiles, size)
    if cache_key in _SMILES_IMG_CACHE:
        _SMILES_IMG_CACHE.move_to_end(cache_key)
        return _SMILES_IMG_CACHE[cache_key]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    src = f"data:image/png;base64,{encoded}"

    _SMILES_IMG_CACHE[cache_key] = src
    if len(_SMILES_IMG_CACHE) > _SMILES_IMG_CACHE_MAX:
        _SMILES_IMG_CACHE.popitem(last=False)

    return src


def _drop_fully_empty_rows(df):
    """Drop rows and columns where every cell is NaN/None or an empty/whitespace string."""
    if df is None or df.empty:
        return df

    empty_like_mask = df.isna().copy()
    for col in df.columns:
        empty_like_mask[col] |= df[col].astype(str).str.strip().eq("")

    return df.loc[
        ~empty_like_mask.all(axis=1),
        ~empty_like_mask.all(axis=0),
    ].copy()


def _build_traceback_alert(action_label, e):
    """Create a reusable error alert with expandable traceback details."""
    error_type = type(e).__name__
    error_trace = traceback.format_exc()
    print(f"ScopeBO run failed with {error_type}: {e}", flush=True)
    print(error_trace, flush=True)
    return dbc.Alert(
        [
            f"An error occurred while {action_label} ({error_type}): {e}",
            html.Br(),
            html.Details(
                [
                    html.Summary("Show full traceback"),
                    html.Pre(
                        error_trace,
                        style={
                            "whiteSpace": "pre-wrap",
                            "fontSize": "0.8rem",
                            "marginTop": "0.5rem",
                        },
                    ),
                ]
            ),
        ],
        style=DANGER_ALERT_STYLE,
        dismissable=False,
    )


def _objective_dropdown_payload(objectives_dict, current_value="", action=""):
    
    objectives_dict = objectives_dict or []

    options = [
        {
            "label": f"Select objective to {action}",
            "value": f"Select objective to {action}",
        }
    ]

    options.extend(
        {
            "label": obj["name"],
            "value": obj["name"],
        }
        for obj in objectives_dict
        if obj.get("name", "").strip()
    )

    option_values = [opt["value"] for opt in options]

    selected_value = (
        current_value
        if current_value in option_values
        else f"Select objective to {action}"
    )

    return options, selected_value


def _search_space_signature(search_space_data):
    """Return a stable signature for one stored search-space payload."""
    if not search_space_data:
        return None
    return hashlib.sha256(search_space_data.encode("utf-8")).hexdigest()


def _build_stale_plot_alert(action_label):
    """Return a warning shown when a derived plot no longer matches the current scope."""
    return dbc.Alert(
        f"This {action_label} is not up to date with the current scope. Click the refresh button above to update it.",
        style=WARNING_ALERT_STYLE,
        dismissable=False,
    )


def _render_stale_plot_warning(current_search_space, stored_signature, action_label):
    """Show a warning only when a rendered plot exists and the scope has changed since generation."""
    if not stored_signature:
        return None

    current_signature = _search_space_signature(current_search_space)
    if not current_signature or current_signature == stored_signature:
        return None

    return _build_stale_plot_alert(action_label)


# Helpers for starting page

def _infer_obj_from_space(df_search):
    """Infer the reaction objectives by looking for 'PENDING' values in the search space dataframe."""
    pending_cols = [col for col in df_search.columns if df_search[col].astype(str).str.contains('PENDING', case=False).any()]
    found_objs = True if pending_cols else False
    if not pending_cols:
        pending_cols = ["Objective 1"]  # default to "Objective 1" if no obj columns found
    return pending_cols, found_objs


def _ensure_search_space_layout(df_search, objectives):
    """Helper function to ensure good search space setup for ScopeBO.run()."""
    if 'priority' not in df_search.columns:
        df_search['priority'] = 0  # default priority 0 if no priority column existed
    # Add any missing objective columns and mark unknown outcomes as PENDING.
    missing_objs = [obj for obj in objectives if obj not in df_search.columns]
    if missing_objs:
        df_search = df_search.reindex(columns=list(df_search.columns) + missing_objs,fill_value="PENDING")

    # canonicalize SMILES strings to ensure consistent matching and avoid duplicates
    df_search.index = df_search.index.map(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True) if pd.notna(x) else x)
    # deduplicate the index
    df_search = df_search[~df_search.index.duplicated(keep='first')]

    return df_search


def _parse_space(df_search):
    """Parse search space into UI-relevant groups and status flags."""
    # check for suggested samples (priority == 1)
    list_sugg = df_search.index[df_search['priority'] == 1].to_list()
    # check for alternative samples (priority between 0 and 1)
    list_alt = df_search.index[(df_search['priority'] < 1) & (df_search["priority"] > 0)].to_list()
    # check if there are any experimental results yet
    exp_results = df_search.index[~df_search.astype(str).eq("PENDING").any(axis=1)]
    status_init = True if len(exp_results) == 0 else False

    return list_sugg, list_alt, status_init


# Helpers for sbustrate suggestion tab

def _build_table(smiles_list, objectives, df_search):
    """Builds a table displaying SMILES strings, their structures, and input fields for objectives."""
    
    header = html.Tr(
        [
            html.Th("SMILES", style=TABLE_HEADER_STYLE),
            html.Th("Structure", style=TABLE_HEADER_STYLE),
            *[html.Th(obj, style=TABLE_HEADER_STYLE) for obj in objectives],
        ]
    )

    rows = []

    for i, smiles in enumerate(smiles_list):

        row_bg = "#fafafa" if i % 2 == 0 else "#ffffff"  # alternate row colors for readability

        row = [
            html.Td(
                smiles,
                style={
                    **TABLE_CELL_STYLE,
                    "backgroundColor": row_bg,
                    "fontFamily": "monospace",
                    "whiteSpace": "nowrap",
                }
            ),
            html.Td(
                _smiles_viewer(smiles),
                style={
                    **TABLE_CELL_STYLE,
                    "textAlign": "center",
                    "backgroundColor": row_bg,
                    "width": "140px",
                }
            ),
        ]

        for objective in objectives:
            existing_value = None
            # Pre-fill from stored values so reported results remain visible after re-render.
            if objective in df_search.columns and smiles in df_search.index:
                candidate = df_search.loc[smiles, objective]
                if pd.notna(candidate) and str(candidate).upper() != "PENDING":
                    existing_value = candidate

            row.append(
                html.Td(
                    dcc.Input(
                        # Pattern-matching IDs let callbacks map each cell to (smiles, objective).
                        id={
                            "type": "objective-input",
                            "smiles": smiles,
                            "objective": objective,
                        },
                        type="text",
                        inputMode="numeric",
                        value=existing_value,
                        debounce=True,
                        style={
                            "width": "70px",
                            "boxSizing": "border-box",
                            "padding": "4px",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "textAlign": "center",
                        }
                    ),
                    style={
                        **TABLE_CELL_STYLE,
                        "backgroundColor": row_bg,
                        "textAlign": "center",
                    }
                )
            )

        rows.append(html.Tr(row))

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "width": "100%",
            "fontsize": "14px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.08)",
        },
    )

    return html.Div(
        table,
        style={
            "overflowX": "auto",
            "maxWidth": "100%",
            "overflowY": "auto",
            "maxHeight": "600px",
            "borderRadius": "8px",
            "backgroundColor": "#ffffff",
        },
    )


def _build_other_table(other_rows, objectives, df_search):
    """Builds an editable table for reporting substrates outside the suggested sets."""
    header = html.Tr(
        [
            html.Th("SMILES", style=TABLE_HEADER_STYLE),
            html.Th("Structure", style=TABLE_HEADER_STYLE),
            *[html.Th(obj, style=TABLE_HEADER_STYLE) for obj in objectives],
        ]
    )

    rows = []

    for row_item in other_rows:
        row_id = row_item.get("row_id")
        smiles = (row_item.get("smiles") or "").strip()
        row_bg = "#fafafa" if row_id % 2 == 0 else "#ffffff"
        row = [
            html.Td(
                dcc.Input(
                    id={"type": "other-smiles-input", "row_id": row_id},
                    type="text",
                    value=smiles,
                    placeholder="Enter SMILES",
                    debounce=True,
                    style={
                        "minWidth": "260px",
                        "boxSizing": "border-box",
                        "padding": "4px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                    },
                ),
                style={
                    **TABLE_CELL_STYLE,
                    "backgroundColor": row_bg,
                }
            ),
            html.Td(
                _smiles_viewer(smiles) if smiles else html.Div("Enter SMILES", style={"color": "#6c757d"}),
                style={
                    **TABLE_CELL_STYLE,
                    "textAlign": "center",
                    "backgroundColor": row_bg,
                    "width": "140px",
                }
            ),
        ]

        for objective in objectives:
            existing_value = None
            if smiles and objective in df_search.columns and smiles in df_search.index:
                candidate = df_search.loc[smiles, objective]
                if pd.notna(candidate) and str(candidate).upper() != "PENDING":
                    existing_value = candidate

            row.append(
                html.Td(
                    dcc.Input(
                        id={
                            "type": "other-objective-input",
                            "row_id": row_id,
                            "objective": objective,
                        },
                        type="text",
                        inputMode="numeric",
                        value=existing_value,
                        debounce=True,
                        style={
                            "width": "70px",
                            "boxSizing": "border-box",
                            "padding": "4px",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "textAlign": "center",
                        }
                    ),
                    style={
                        **TABLE_CELL_STYLE,
                        "backgroundColor": row_bg,
                        "textAlign": "center",
                    }
                )
            )

        rows.append(html.Tr(row))

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "width": "100%",
            "fontsize": "14px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.08)",
        },
    )

    return html.Div([
        dbc.Button(
            "Add another substrate",
            id="btn-add-other-row",
            color="secondary",
            size="sm",
            className="mb-2",
        ),
        html.Div(
            table,
            style={
                "overflowX": "auto",
                "maxWidth": "100%",
                "overflowY": "auto",
                "maxHeight": "600px",
                "borderRadius": "8px",
                "backgroundColor": "#ffffff",
            },
        ),
    ])


def _update_search_space_from_report(df_search, input_ids, input_values, objective_names):
    """
    Merge completed reporting-table values back into the search space dataframe.
    Helper function for the report_results callback.
    """
    if not objective_names:
        return df_search, 0, 0, 0

    # Re-group flat pattern-matching inputs into row-wise records keyed by SMILES.
    values_by_smiles = {}
    for field_id, value in zip(input_ids, input_values):
        smiles = field_id.get("smiles")
        objective = field_id.get("objective")
        if smiles not in values_by_smiles:
            values_by_smiles[smiles] = {}
        values_by_smiles[smiles][objective] = value

    updated_rows = 0
    skipped_rows = 0
    invalid_numeric_count = 0

    for smiles, objective_values in values_by_smiles.items():
        # A row is reportable only if every objective field is present and non-empty.
        has_all_values = all(
            pd.notna(objective_values.get(objective))
            and str(objective_values.get(objective)).strip() != ""
            for objective in objective_names
        )
        if not has_all_values:
            skipped_rows += 1
            continue

        # Numeric validation is explicit because inputs are text to allow warning on bad entries.
        numeric_values = {}
        row_has_invalid_numeric = False
        for objective in objective_names:
            raw_value = objective_values.get(objective)
            if isinstance(raw_value, str):
                raw_value = raw_value.strip()
            try:
                numeric_values[objective] = float(raw_value)
            except (TypeError, ValueError):
                invalid_numeric_count += 1
                row_has_invalid_numeric = True

        if row_has_invalid_numeric:
            skipped_rows += 1
            continue

        row_changed = False
        for objective in objective_names:
            old_value = df_search.loc[smiles, objective] if objective in df_search.columns else "PENDING"
            try:
                old_numeric = float(old_value)
                if old_numeric != numeric_values[objective]:
                    row_changed = True
            except (TypeError, ValueError):
                # Treat PENDING/non-numeric existing values as different from valid numeric input.
                row_changed = True

            df_search.loc[smiles, objective] = numeric_values[objective]

        if row_changed:
            updated_rows += 1

    return df_search, updated_rows, skipped_rows, invalid_numeric_count


def _update_search_space_from_other_report(
    df_search,
    other_rows,
    input_ids,
    input_values,
    objective_names,
):
    """Merge reported values from the editable 'other substrates' table into the search space."""
    if not objective_names:
        return df_search, 0, 0, 0, 0

    row_to_smiles = {row.get("row_id"): (row.get("smiles") or "").strip() for row in other_rows}
    values_by_row = {}
    for field_id, value in zip(input_ids, input_values):
        row_id = field_id.get("row_id")
        objective = field_id.get("objective")
        if row_id not in values_by_row:
            values_by_row[row_id] = {}
        values_by_row[row_id][objective] = value

    updated_rows = 0
    skipped_rows = 0
    invalid_numeric_count = 0
    missing_smiles_count = 0

    for row_id, objective_values in values_by_row.items():
        smiles = row_to_smiles.get(row_id, "")
        if not smiles:
            skipped_rows += 1
            continue

        # canonicalize the SMILES for matching
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)

        # "Other" mode is for reporting existing search-space entries outside the current suggestion sets.
        if smiles not in df_search.index:
            missing_smiles_count += 1
            skipped_rows += 1
            continue

        has_all_values = all(
            pd.notna(objective_values.get(objective))
            and str(objective_values.get(objective)).strip() != ""
            for objective in objective_names
        )
        if not has_all_values:
            skipped_rows += 1
            continue

        numeric_values = {}
        row_has_invalid_numeric = False
        for objective in objective_names:
            raw_value = objective_values.get(objective)
            if isinstance(raw_value, str):
                raw_value = raw_value.strip()
            try:
                numeric_values[objective] = float(raw_value)
            except (TypeError, ValueError):
                invalid_numeric_count += 1
                row_has_invalid_numeric = True

        if row_has_invalid_numeric:
            skipped_rows += 1
            continue

        row_changed = False
        for objective in objective_names:
            old_value = df_search.loc[smiles, objective] if objective in df_search.columns else "PENDING"
            try:
                old_numeric = float(old_value)
                if old_numeric != numeric_values[objective]:
                    row_changed = True
            except (TypeError, ValueError):
                row_changed = True

            df_search.loc[smiles, objective] = numeric_values[objective]

        if row_changed:
            updated_rows += 1

    return df_search, updated_rows, skipped_rows, invalid_numeric_count, missing_smiles_count


# Helpers for UMAP tab

def _build_umap_graph_children(figure_dict):
    """Rebuild the stored UMAP figure into the tab content."""
    if not figure_dict:
        return None

    graph = dcc.Graph(
        id="graph-umap",
        figure=go.Figure(figure_dict),
        clear_on_unhover=True,
        config={"displaylogo": False, "responsive": True},
    )

    return html.Div([graph, dcc.Tooltip(id="tooltip-umap")])


# Helpers for SHAP tab

def _build_shap_graph_children(beeswarm_figure_dict, bar_figure_dict):
    """Rebuild stored SHAP figures into the tab content."""
    if not beeswarm_figure_dict or not bar_figure_dict:
        return None

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "?",
                        id="help-shap-beeswarm",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "20px",
                            "height": "20px",
                            "borderRadius": "50%",
                            "backgroundColor": "#0d6efd",
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "12px",
                            "cursor": "help",
                        },
                    ),
                    dbc.Tooltip(
                        [
                            "Shows the impact of the feature values on the model output for each sample. ",
                            "The color corresponds to the normalized feature value. ",
                            "A higher SHAP value indicates a greater contribution to the model's prediction.",
                        ],
                        target="help-shap-beeswarm",
                        placement="right",
                    ),
                ],
                style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "0.35rem"},
            ),
            dcc.Graph(
                id="graph-shap-beeswarm",
                figure=go.Figure(beeswarm_figure_dict),
                clear_on_unhover=True,
                config={"displaylogo": False, "responsive": True},
            ),
            dcc.Tooltip(id="tooltip-shap-beeswarm"),
            html.Br(),
            html.Div(
                [
                    html.Span(
                        "?",
                        id="help-shap-bar",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "20px",
                            "height": "20px",
                            "borderRadius": "50%",
                            "backgroundColor": "#0d6efd",
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "12px",
                            "cursor": "help",
                        },
                    ),
                    dbc.Tooltip(
                        [
                            "Show the average impact of each feature on the model output across all evaluated samples. ",
                            "The impact does however not indicate if the impact is positive or negative.",
                        ],
                        target="help-shap-bar",
                        placement="right",
                    ),
                ],
                style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "0.35rem"},
            ),
            dcc.Graph(
                id="graph-shap-bar",
                figure=go.Figure(bar_figure_dict),
                config={"displaylogo": False, "responsive": True},
            ),
        ]
    )


# Helpers for prediction tab

def _build_pred_graph_children(figure_dict):
    """Rebuild the stored prediction figure into the tab content."""
    if not figure_dict:
        return None

    graph = dcc.Graph(
        id="graph-pred",
        figure=go.Figure(figure_dict),
        clear_on_unhover=True,
        config={"displaylogo": False, "responsive": True},
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "?",
                        id="help-pred-umap",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "20px",
                            "height": "20px",
                            "borderRadius": "50%",
                            "backgroundColor": "#0d6efd",
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "12px",
                            "cursor": "help",
                        },
                    ),
                    dbc.Tooltip(
                        [
                            "Chemical space visualization of the values predicted by a Gaussian process regressor (ScopeBO surrogate model). ",
                            "Larger markers indicate lower uncertainty in the prediction. ",
                        ],
                        target="help-pred-umap",
                        placement="right",
                    ),
                ],
                style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "0.35rem"},
            ),
            graph,
            dcc.Tooltip(id="tooltip-pred"),
        ]
    )





# Helpers for compound featurization

def _append_featurization_store(existing_data, new_df_json, source):
    """Append one feature dataframe JSON to store payload as a list of dict entries."""
    existing_list = existing_data or []
    if new_df_json is None:
        return existing_list
    return [*existing_list, {"data": new_df_json, "source": source}]

def _extract_smiles_from_upload(upload_contents, upload_filename):
    """Parse uploaded smiles CSV and return (smiles_list, error_alert)."""
    if upload_contents is None:
        return None, dbc.Alert(
            "Please upload a SMILES CSV before featurization.",
            style=WARNING_ALERT_STYLE,
            dismissable=False,
        )

    if not upload_filename or not upload_filename.lower().endswith(".csv"):
        return None, dbc.Alert(
            "Please upload a valid CSV file with a 'smiles' column.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    try:
        _, content_string = upload_contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df_input = pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0)
        df_input = _drop_fully_empty_rows(df_input)
    except Exception:
        return None, dbc.Alert(
            "Could not read the uploaded CSV file.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    if df_input.empty:
        return None, dbc.Alert(
            "Uploaded SMILES CSV is empty.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    smiles_columns = [col for col in df_input.columns if str(col).strip().lower() == "smiles"]
    if not smiles_columns:
        return None, dbc.Alert(
            "The uploaded CSV must contain a column named 'smiles'.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )

    smiles_col = smiles_columns[0]
    smiles_list = (
        df_input[smiles_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    smiles_list = [s for s in smiles_list if s]
    if not smiles_list:
        return None, dbc.Alert(
            "No valid SMILES strings found in the uploaded file.",
            style=DANGER_ALERT_STYLE,
            dismissable=False,
        )
    
    # Check for invalid SMILES
    invalid = [
        smi
        for smi in smiles_list
        if Chem.MolFromSmiles(smi) is None
    ]
    if invalid:
        return (
            None,
            dbc.Alert(
                f"Found invalid SMILES "
                f"(first invalid: '{invalid[0]}'). Please fix the input file and try again.",
                style=DANGER_ALERT_STYLE,
                dismissable=False,
            ),
        )

    return smiles_list, None


def _set_featurization_job_state(run_id, **kwargs):
    """Safely update in-memory state for one featurization job."""
    with _FEATURIZATION_JOBS_LOCK:
        job_state = _FEATURIZATION_JOBS.setdefault(run_id, {})
        job_state.update(kwargs)


def _get_featurization_job_state(run_id):
    """Safely read in-memory state for one featurization job."""
    with _FEATURIZATION_JOBS_LOCK:
        return dict(_FEATURIZATION_JOBS.get(run_id, {}))


def _has_active_featurization_job():
    """Return True if any featurization job is still running."""
    with _FEATURIZATION_JOBS_LOCK:
        for state in _FEATURIZATION_JOBS.values():
            if state.get("status") == "running":
                return True
    return False


def _is_featurization_job_running(run_id):
    """Return True when the specified featurization job is still running."""
    if not run_id:
        return False
    state = _get_featurization_job_state(run_id)
    return state.get("status") == "running"


def _run_featurization_job(run_id, smiles_list):
    """Worker function executed in a background thread."""
    total = len(smiles_list)

    def _progress(done_count, total_count, message):
        denominator = max(int(total_count or 0), 1)
        percent = int(round((float(done_count) / float(denominator)) * 100))
        percent = max(0, min(100, percent))
        _set_featurization_job_state(
            run_id,
            status="running",
            done=int(done_count),
            total=int(total_count),
            percent=percent,
            message=message,
        )

    _set_featurization_job_state(
        run_id,
        status="running",
        done=0,
        total=total,
        percent=0,
        message="Queued featurization job...",
        cancel_requested=False,
    )

    try:
        def _check_cancel_requested():
            return bool(_get_featurization_job_state(run_id).get("cancel_requested", False))

        _, df_descriptors = calculate_morfeus_descriptors_web(
            smiles_list=smiles_list,
            filename="unused_webapp_output.csv",
            progress_callback=_progress,
            check_cancel=_check_cancel_requested,
        )
        descriptors_json = df_descriptors.to_json(date_format="iso", orient="split")
        _set_featurization_job_state(
            run_id,
            status="done",
            done=total,
            total=total,
            percent=100,
            message="Featurization finished",
            descriptors_json=descriptors_json,
            row_count=len(df_descriptors),
            cancel_requested=False,
            result_emitted=False,
        )
    except Exception as e:
        interrupted = "interrupted by user" in str(e).lower()
        if interrupted:
            current_state = _get_featurization_job_state(run_id)
            _set_featurization_job_state(
                run_id,
                status="cancelled",
                done=int(current_state.get("done", 0)),
                total=int(current_state.get("total", total)),
                percent=int(current_state.get("percent", 0)),
                message="Featurization interrupted by user.",
                cancel_requested=False,
                result_emitted=False,
            )
            return

        _set_featurization_job_state(
            run_id,
            status="error",
            done=0,
            total=total,
            percent=0,
            message=str(e),
            error=_build_traceback_alert("running featurization", e),
            cancel_requested=False,
            result_emitted=False,
        )


# Helpers for preprocessing

def _normalize_store(store):
    """Ensure that each item in the featurization data store has a display name and an include flag."""
    for i, item in enumerate(store):
        item.setdefault(
            "display_name",
            item.get("filename", f"Dataset {i+1}")
        )
        item.setdefault("include", True)
    return store


############################################################
# APP EXECUTION
############################################################

# # --- EXECUTE APPLICATION SERVER ---
# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=True)
    

# Function to check if the port is actively listening
def is_server_ready(host="127.0.0.1", port=8050):
    try:
        # Try to make a quick socket connection to the port
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False

# Background thread function that waits for the server
def open_browser():
    # Only run this if we aren't in the Werkzeug reloader master process (if debug=True)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        while True:
            if is_server_ready():
                webbrowser.open_new("http://127.0.0.1:8050/")
                break
            time.sleep(0.2)  # Check every 200 milliseconds

if __name__ == "__main__":
    # 'daemon=True' ensures this thread dies automatically if the main app crashes
    Thread(target=open_browser, daemon=True).start()
    
    # Run the server
    app.run(debug=False, port=8050)