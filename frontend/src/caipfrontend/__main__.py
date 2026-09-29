"""Nicegui frontend module"""

from nicegui import app, ui

from caipfrontend.session import session
from caipfrontend.pages.prediction import prediction_page as root

app.on_shutdown(session.aclose)


def main():
    ui.run(host="127.0.0.1", port=8081, reload=False, title="AI Pathology", root=root)


if __name__ == "__main__":
    main()
