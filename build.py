import re
from pathlib import Path

def get_requirements():
    """
    Read and process requirements from requirements.txt with flexible version handling.
    Converts exact versions (== or ===) to minimum versions (>=).
    Handles inline comments and cleans up formatting.
    """
    try:
        requirements = Path('requirements.txt').read_text().splitlines()
    except FileNotFoundError:
        print("Warning: requirements.txt not found. No dependencies will be included.")
        return []
        
    processed_requirements = []
    
    for req in requirements:
        req = req.strip()
        
        # Skip empty lines and comments
        if not req or req.startswith('#'):
            continue
            
        try:
            # Handle inline comments
            req = req.split('#')[0].strip()
            
            # Extract package name and version if they exist
            if re.search(r'[=<>]', req):
                pkg_name = re.split(r'[=<>]+', req)[0]
                version_match = re.search(r'(?:={2,3}|>=?|<=?)([^#\s]+)', req)
                if version_match:
                    version = version_match.group(1)
                    # Convert to flexible version requirement
                    req = f"{pkg_name}>={version}"
            
            processed_requirements.append(req)
            
        except Exception as e:
            print(f"Warning: Could not process requirement '{req}': {str(e)}")
            continue
    
    return processed_requirements

def build(setup_kwargs):
    """
    Updates build arguments with processed requirements.
    This function is called by the build system.
    """
    requirements = get_requirements()
    setup_kwargs.update({
        "install_requires": requirements,
    })
    
    # Print processed requirements for verification
    if requirements:
        print("\nProcessed requirements:")
        for req in requirements:
            print(f"  - {req}")