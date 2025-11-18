from mmengine.registry import TRANSFORMS as MM
from mmaction.registry import TRANSFORMS as MA
import pprint

print('mmengine TRANSFORMS count =', len(MM.module_dict))
print('mmaction TRANSFORMS count =', len(MA.module_dict))
print('mmengine has PreNormalize2D?', 'PreNormalize2D' in MM.module_dict)
print('mmaction has PreNormalize2D?', 'PreNormalize2D' in MA.module_dict)
print()
pprint.pprint(list(MA.module_dict.keys())[:200])
