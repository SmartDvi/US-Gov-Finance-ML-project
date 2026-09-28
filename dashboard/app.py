"""
Multi-page Dash dashboard for the US state government finance ML pipeline.

    uv run python -m dashboard.app        ->  http://127.0.0.1:8050

Each file in dashboard/pages/ registers one page (Dash "pages" feature).
This file only builds the shell around them: header, navigation sidebar
(generated from the page registry) and the area where pages render.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `dashboard` and `src` importable however the app is started:
# `python -m dashboard.app` from the project root, or `python app.py` inside dashboard/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dash  # noqa: E402  (imports below need the path set up above)
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback

from dashboard.data import CONFIG

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    title="US State Finance ML",
    suppress_callback_exceptions=True,
)


def navigation() -> dmc.Stack:
    """One NavLink per registered page, in the order set by each page."""
    return dmc.Stack(
        [
            dmc.NavLink(
                id={"type": "nav", "path": page["path"]},
                label=page["name"],
                description=page.get("description"),
                href=page["path"],
            )
            for page in sorted(dash.page_registry.values(), key=lambda p: p["order"])
        ],
        gap=4,
    )


header = dmc.Group(
    [
        dmc.Group(
            [
                dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                dmc.Title("US State Finance ML", order=3),
            ],
            gap="sm",
        ),
        dmc.Badge("PySpark · MLflow", variant="light", visibleFrom="xs"),
    ],
    justify="space-between",
    h="100%",
    px="md",
)

app.layout = dmc.MantineProvider(
    dmc.AppShell(
        [
            dmc.AppShellHeader(header),
            dmc.AppShellNavbar(navigation(), p="md"),
            dmc.AppShellMain(dash.page_container),
        ],
        id="appshell",
        header={"height": 60},
        navbar={"width": 260, "breakpoint": "sm", "collapsed": {"mobile": True}},
        padding="md",
    ),
    theme={"primaryColor": "indigo", "defaultRadius": "md"},
)


@callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def toggle_mobile_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


@callback(
    Output({"type": "nav", "path": dash.ALL}, "active"),
    Input("_pages_location", "pathname"),
)
def highlight_current_page(pathname):
    return [
        page["path"] == pathname
        for page in sorted(dash.page_registry.values(), key=lambda p: p["order"])
    ]


if __name__ == "__main__":
    app.run(host=CONFIG["dashboard"]["host"], port=CONFIG["dashboard"]["port"], debug=False)
