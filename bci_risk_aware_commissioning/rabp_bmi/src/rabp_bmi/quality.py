from __future__ import annotations


# Commonly excluded in PhysioNet EEGMMIDB left/right MI studies because of
# damaged recordings, sampling-rate mismatch, incomplete annotations, or too few
# usable samples. We keep this explicit so sensitivity analyses can include them.
DEFAULT_EXCLUDED_SUBJECTS = frozenset({"S088", "S092", "S100", "S104"})

