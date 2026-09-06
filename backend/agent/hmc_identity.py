"""Pure HMC field syntax; neither source facts nor tool permissions."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Mapping


# Compact HMC parsing must have an unambiguous component boundary.  Making
# every delimiter optional lets any bare eight-digit mission/batch number be
# split as ``6 digit LCN + 1 digit HAMC + 1 digit HSC``.  A wildcard itself is
# an unambiguous boundary and therefore remains usable with or without spaces;
# an alphanumeric HAMC requires explicit punctuation on both sides.
_HMC_COMPACT_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])(?P<lcn>\d{6,10})\s*"
        r"(?P<hamc>\*\*)\s*(?P<hsc>\d{1,6})(?![A-Za-z0-9])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?P<lcn>\d{6,10})\s*[-_/]\s*"
        r"(?P<hamc>\*\*|[A-Za-z0-9]{1,4})\s*[-_/]\s*"
        r"(?P<hsc>\d{1,6})(?![A-Za-z0-9])"
    ),
)
_HMC_FIELD_LABEL = re.compile(
    r"(?<![A-Za-z0-9_])(?:LCN|HAMC|HSC)(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
_HMC_FIELD_PATTERNS = {
    "lcn": re.compile(
        r"(?<![A-Za-z0-9_])LCN\s*(?:[:=]\s*)?(\d{6,10})(?!\d)",
        flags=re.IGNORECASE,
    ),
    "hamc": re.compile(
        r"(?<![A-Za-z0-9_])HAMC\s*(?:[:=]\s*)?"
        r"(?!LCN\b|HAMC\b|HSC\b)(\*\*|[A-Za-z0-9]{1,4})(?![A-Za-z0-9*])",
        flags=re.IGNORECASE,
    ),
    "hsc": re.compile(
        r"(?<![A-Za-z0-9_])HSC\s*(?:[:=]\s*)?(\d{1,6})(?!\d)",
        flags=re.IGNORECASE,
    ),
}


def extract_hmc_components(text: Any) -> Dict[str, str]:
    """Parse labelled HMC fields or a complete compact LCN/HAMC/HSC code.

    Labelled ``LCN + HSC`` is intentionally retained without inventing a
    missing HAMC.  Once any field label is present, compact parsing is not
    attempted: otherwise ``LCN 91810001 HSC 0001`` is misread as if the word
    ``HSC`` were the HAMC value.
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("—", "-").replace("–", "-")
    labelled: Dict[str, str] = {}
    for key, pattern in _HMC_FIELD_PATTERNS.items():
        match = pattern.search(value)
        if match:
            labelled[key] = match.group(1).upper()

    if "hsc" in labelled and labelled["hsc"].isdigit() and len(labelled["hsc"]) < 4:
        labelled["hsc"] = labelled["hsc"].zfill(4)
    if "lcn" in labelled and "hsc" in labelled:
        if "hamc" not in labelled:
            return labelled
        labelled["hmc_code"] = f"{labelled['lcn']}{labelled['hamc']}{labelled['hsc']}"
        return labelled

    # A label makes the input field-oriented.  Fail closed instead of letting
    # the permissive compact expression consume another label as a code part.
    if _HMC_FIELD_LABEL.search(value):
        return {}

    match = None
    for pattern in _HMC_COMPACT_PATTERNS:
        match = pattern.search(value)
        if match:
            break
    if not match:
        return {}
    result = {key: item.upper() for key, item in match.groupdict().items()}
    if result["hsc"].isdigit() and len(result["hsc"]) < 4:
        result["hsc"] = result["hsc"].zfill(4)
    result["hmc_code"] = f"{result['lcn']}{result['hamc']}{result['hsc']}"
    return result


def normalize_typed_hmc_components(value: Any) -> Dict[str, str]:
    """Validate an already typed HMC component object.

    This deliberately accepts a mapping only.  Planner search terms and other
    free-form strings are retrieval hints; their digit shapes must never be
    promoted into an LCN/HSC identity.  The validated result uses the same
    canonical formatting as :func:`extract_hmc_components`.
    """

    if not isinstance(value, Mapping):
        return {}
    lcn = unicodedata.normalize("NFKC", str(value.get("lcn") or "")).strip()
    hamc = unicodedata.normalize("NFKC", str(value.get("hamc") or "")).strip()
    hsc = unicodedata.normalize("NFKC", str(value.get("hsc") or "")).strip()
    if not lcn or not hsc:
        return {}
    rendered = f"LCN {lcn} "
    if hamc:
        rendered += f"HAMC {hamc} "
    rendered += f"HSC {hsc}"
    parsed = extract_hmc_components(rendered)
    if not parsed:
        return {}
    # Round-trip equality prevents permissive text parsing from accepting
    # trailing prose or a partially consumed typed value.
    expected = {
        "lcn": lcn.upper(),
        "hsc": hsc.zfill(4) if hsc.isdigit() and len(hsc) < 4 else hsc.upper(),
    }
    if hamc:
        expected["hamc"] = hamc.upper()
    if any(parsed.get(key) != item for key, item in expected.items()):
        return {}
    return parsed

