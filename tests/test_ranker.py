import pytest

from core.ranker import SearchEngine


@pytest.fixture
def search_engine() -> SearchEngine:
    engine = SearchEngine()
    engine.build_index({
        1: ('Doc1', 'database systems and network design'),
        2: ('Doc2', 'network security and protocols'),
        3: ('Doc3', 'database architecture and storage')
    })
    return engine


def test_boolean_and_retrieves_exact_matches(search_engine: SearchEngine) -> None:
    results = search_engine.search('database AND network')
    assert len(results) == 1
    assert results[0]['doc_id'] == 1


def test_boolean_or_retrieves_union_of_matches(search_engine: SearchEngine) -> None:
    results = search_engine.search('database OR security')
    assert {result['doc_id'] for result in results} == {1, 2, 3}
    assert len(results) == 3


def test_vector_space_model_search_still_ranks_documents(search_engine: SearchEngine) -> None:
    results = search_engine.search('network database')
    assert len(results) == 3
    assert results[0]['doc_id'] == 1
