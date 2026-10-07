from tools.filesystem import list_files
from tools.code_search import search_code

def test_list_and_search():
    files=list_files('sample_repo')
    assert 'app/main.py' in files
    assert search_code('sample_repo','FastAPI')
