# pytest test_rdf_explorer.py -v

import pytest
from server import triplestore_lifespan, get_mode, FastMCP, Context
from unittest.mock import AsyncMock, patch
import rdflib

# Mock fixtures (unchanged)
@pytest.fixture
def mock_mcp():
    mcp = FastMCP("Test")
    mcp._lifespan_context = {
        "active_external_endpoint": None,
        "triple_file": "test.ttl",
        "sparql_endpoint": ""
    }
    return mcp

@pytest.fixture
def mock_context(mock_mcp):
    class MockRequestContext:
        lifespan_context = mock_mcp._lifespan_context
    class MockContext:
        request_context = MockRequestContext()
    return MockContext()

# Existing tests (unchanged)
def test_get_mode_local(mock_context):
    result = get_mode(mock_context)
    assert result == "Local File Mode with Dataset: 'test.ttl'"

def test_get_mode_sparql(mock_context):
    mock_context.request_context.lifespan_context["active_external_endpoint"] = "https://dbpedia.org/sparql"
    mock_context.request_context.lifespan_context["sparql_endpoint"] = "https://dbpedia.org/sparql"
    result = get_mode(mock_context)
    assert result == "SPARQL Endpoint Mode with Endpoint: 'https://dbpedia.org/sparql'"

@pytest.mark.asyncio
async def test_triplestore_lifespan_local():
    mock_server = AsyncMock()
    with patch("rdflib.Graph.parse") as mock_parse:
        mock_parse.return_value = rdflib.Graph()
        async with triplestore_lifespan(mock_server, "test.ttl", "") as context:
            assert context["triple_file"] == "test.ttl"
            assert context["active_external_endpoint"] is None
            assert isinstance(context["graph"], rdflib.Graph)

@pytest.mark.asyncio
async def test_triplestore_lifespan_sparql():
    mock_server = AsyncMock()
    with patch("rdflib.plugins.stores.sparqlstore.SPARQLStore.query") as mock_query:
        mock_query.return_value = []
        async with triplestore_lifespan(mock_server, "test.ttl", "https://dbpedia.org/sparql") as context:
            assert context["sparql_endpoint"] == "https://dbpedia.org/sparql"
            assert context["active_external_endpoint"] == "https://dbpedia.org/sparql"
            assert "graph" in context

# New failure tests
@pytest.mark.asyncio
async def test_triplestore_lifespan_sparql_failure():
    mock_server = AsyncMock()
    with patch("rdflib.plugins.stores.sparqlstore.SPARQLStore.query") as mock_query:
        mock_query.side_effect = Exception("Connection failed")
        with pytest.raises(Exception, match="Connection failed"):
            async with triplestore_lifespan(mock_server, "test.ttl", "http://invalid.endpoint"):
                pass

@pytest.mark.asyncio
async def test_triplestore_lifespan_local_file_missing():
    mock_server = AsyncMock()
    with patch("rdflib.Graph.parse") as mock_parse:
        mock_parse.side_effect = FileNotFoundError("File not found")
        with pytest.raises(FileNotFoundError, match="File not found"):
            async with triplestore_lifespan(mock_server, "missing.ttl", ""):
                pass