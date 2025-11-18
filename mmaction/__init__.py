# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.7.1'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (digit_version(mmcv_minimum_version) <= mmcv_version
        < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (digit_version(mmengine_minimum_version) <= mmengine_version
        < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__']

# Ensure mmaction registries and transforms are imported when the package
# is imported so that mmengine's global registries can discover custom
# transforms and other components. This is a best-effort import that
# should not break importing mmaction if something goes wrong.
try:
    # Import registry to run any registration side-effects there
    from . import registry  # noqa: F401

    # Import transforms module (module-level decorators/register calls)
    # so that custom transforms (PreNormalize2D, AddGaussianNoise, etc.)
    # are registered into the local mmaction registry. This import
    # also helps in making them visible before mmengine.Compose builds
    # pipelines that reference them.
    from .datasets import transforms as transforms  # noqa: F401
    # After importing, ensure entries from mmaction.registry.TRANSFORMS
    # are copied into mmengine's global TRANSFORMS registry. We do a
    # direct copy of module_dict as a robust fallback for mmengine
    # versions where register_module invocation may behave differently.
    try:
        from . import registry as _local_registry
        from mmengine.registry import TRANSFORMS as _mmengine_transforms

        for _k, _v in getattr(_local_registry.TRANSFORMS, 'module_dict', {}).items():
            if _k in getattr(_mmengine_transforms, 'module_dict', {}):
                continue
            try:
                _mmengine_transforms.module_dict[_k] = _v
            except Exception:
                # best-effort: ignore entries that cannot be copied
                pass
    except Exception:
        # don't fail package import if copying fails
        pass
except Exception as _err:  # pragma: no cover - import-time best-effort
    import warnings

    warnings.warn(
        f"mmaction import-time registration warning: failed to import"
        f" registries/transforms: {_err}")
