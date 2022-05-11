import sys
from pathlib import Path

PROJECT_SCIPERS = 'Proj_302882_335482_343955'

def load_path(mini_project):
    """"Gets the path to bestmodel.pth.
    May be better ways of achieving this 
    but does the job."""
    RELATIVE_PATH = Path("")
    for p in sys.path: # Look through sys paths
        if PROJECT_SCIPERS in p:
            # Should always be a path in sys.path that contains 'Proj_302882_335482_343955'
            RELATIVE_PATH = Path(p)
            break
    BESTMODEL = f"bestmodel.pth"
    MINI_PROJ_SUB_PATH = f"Miniproject_"+str(mini_project)
    PATH_1 = RELATIVE_PATH / BESTMODEL # used if sys is inside miniproject 1 directory
    PATH_2 = RELATIVE_PATH / MINI_PROJ_SUB_PATH / BESTMODEL # used if sys is inside Proj_302... directory
    return PATH_1 if PATH_1.exists() else PATH_2