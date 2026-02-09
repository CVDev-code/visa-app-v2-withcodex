"""
Index builder for final packet.
"""

from weasyprint import HTML


CRITERION_SCHEDULE_MAP = {
    "1": "Schedule 3: Awards",
    "2_past": "Schedule 4: Distinguished Productions",
    "2_future": "Schedule 4: Distinguished Productions",
    "3": "Schedule 5: Recognition (Reviews)",
    "4_past": "Schedule 6: Distinguished Organizations",
    "4_future": "Schedule 6: Distinguished Organizations",
    "5": "Schedule 7: Commercial/Critical Success",
    "6": "Schedule 5: Recognition (Expert/Org)",
    "7": "Schedule 8: High Salary"
}


def schedule_for_criterion(criterion_id: str) -> str:
    return CRITERION_SCHEDULE_MAP.get(criterion_id, "Schedule: Other")


def build_index_pdf(entries, packet_title="Index") -> bytes:
    """
    entries: list of dicts {label: str, page: int}
    """
    rows = "\n".join(
        f"<tr><td>{e['label']}</td><td style='text-align:right'>p{e['page']}</td></tr>"
        for e in entries
    )

    html = f"""
    <html>
      <body>
        <h1 style="font-family: Arial;">{packet_title}</h1>
        <table style="width:100%; font-family: Arial; font-size: 12px;">
          <tbody>
            {rows}
          </tbody>
        </table>
      </body>
    </html>
    """
    return HTML(string=html).write_pdf()

