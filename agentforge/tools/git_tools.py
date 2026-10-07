from tools.commands import run_command

def diff(root: str) -> str:
    return run_command(['git','diff'],root,30).get('stdout','')
