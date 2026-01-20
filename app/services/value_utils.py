from datetime import datetime, timezone


def get_value(field_value):
    """
    OCI may return:
    - real responses: .text
    - unit tests (mock): .value
    This helper safely supports both.
    """
    if not field_value:
        return None

    return getattr(field_value, "text", getattr(field_value, "value", None))


def amount_format(value):
    """
    Converts monetary values to float.
    Examples:
      "$58.11" -> 58.11
      "4,293.55" -> 4293.55
    """
    if value in (None, ""):
        return None

    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def format_date(date_text):
    """
    Converts:
      'Mar 06 2012' -> '2012-03-06T00:00:00+00:00'
    Leaves ISO strings untouched.
    """
    if not date_text:
        return None

    try:
        # Already ISO?
        if "T" in date_text:
            return date_text

        dt = datetime.strptime(date_text.strip(), "%b %d %Y")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return date_text
