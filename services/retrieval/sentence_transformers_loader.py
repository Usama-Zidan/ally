"""
The single import point for sentence-transformers, plus a workaround for
Windows hosts whose Application Control policy blocks pyarrow's native
extension.

sentence-transformers 6.x imports ``datasets`` at module scope, and
``datasets`` imports ``pyarrow.dataset``, which loads pyarrow's native
``_dataset`` extension. Where a policy blocks that extension the whole chain
fails with

    ImportError: The pyarrow installation is not built with support for
    'dataset' (DLL load failed while importing _dataset: an Application
    Control policy has blocked this file)

and every embedding, indexing and reranking call dies with it. Nothing is
misconfigured and the message misleads: the OS refuses to load the DLL, so
retrying, reinstalling sentence-transformers, or pinning another version of
it changes nothing. Temporal retries the activity and fails identically.

``datasets`` only serves fine-tuning, which this service never does, so when
the real extension will not load we register placeholder modules for
``pyarrow.dataset`` and ``pyarrow._dataset`` before the import chain reaches
them. ``datasets`` then finishes importing: everything it wants from
``pyarrow.dataset`` at import time is a type annotation, and the code that
genuinely queries it (scanning datasets off disk) is never on our path.

Where the extension loads normally — Linux CI, Docker, unblocked Windows —
the real module is imported and used, so this is a no-op and nothing else
about runtime behavior changes.
"""
from __future__ import annotations

import importlib
import sys
import types
from functools import lru_cache

import structlog

log = structlog.get_logger()

STUBBED_MODULES = ("pyarrow._dataset", "pyarrow.dataset")


class _PlaceholderModule(types.ModuleType):
    """Stands in for a module whose native extension the OS refuses to load.

    Every attribute read invents a throwaway class, because ``datasets``
    references ``pyarrow.dataset`` names in annotations it evaluates eagerly
    (``filters: Optional[Union[pds.Expression, list[tuple]]]``) and each of
    those has to be a real class. Dunder lookups still raise, so nothing that
    probes a module for capabilities is fooled.
    """

    def __getattr__(self, name: str) -> type:
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (), {"__module__": self.__name__})
        setattr(self, name, placeholder)
        return placeholder


def _install_pyarrow_dataset_stub() -> None:
    """Register the placeholder modules for the rest of this process."""
    for name in STUBBED_MODULES:
        stub = _PlaceholderModule(name)
        # __path__ marks the placeholder as a package, so submodule lookups
        # under pyarrow.dataset resolve to "not found" instead of failing on
        # a missing attribute.
        stub.__path__ = []
        sys.modules[name] = stub

    pyarrow = importlib.import_module("pyarrow")
    for name in STUBBED_MODULES:
        # ``import pyarrow.dataset as pds`` binds the name off the parent
        # package on some Python versions, so register in both places.
        setattr(pyarrow, name.rsplit(".", 1)[-1], sys.modules[name])


def ensure_pyarrow_dataset_importable() -> bool:
    """Make ``pyarrow.dataset`` importable; return True if the real one loaded.

    Returns False when the native extension is blocked and a placeholder is
    standing in for it. That is logged once, so a machine relying on the
    placeholder is visible in worker output rather than looking silently
    degraded.
    """
    if isinstance(sys.modules.get("pyarrow.dataset"), _PlaceholderModule):
        return False
    try:
        importlib.import_module("pyarrow.dataset")
        return True
    except ImportError:
        _install_pyarrow_dataset_stub()
        log.warning(
            "pyarrow_dataset_placeholder_installed",
            detail=(
                "pyarrow's native _dataset extension is blocked by an Application "
                "Control policy; stubbing pyarrow.dataset so sentence-transformers "
                "can import. Inference is unaffected, fine-tuning is unavailable."
            ),
        )
        return False


@lru_cache(maxsize=1)
def load_sentence_transformers() -> types.ModuleType:
    """Import and return the ``sentence_transformers`` module.

    Routed through here rather than imported at module scope so the embedding
    and reranking modules stay cheap to import: torch only loads on the first
    actual model use, once per process, instead of at worker startup.
    """
    ensure_pyarrow_dataset_importable()
    return importlib.import_module("sentence_transformers")
