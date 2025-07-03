import argparse
import asyncio
import json
import logging
import sys
import time
import tiktoken
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Dict, Any

import rdflib
import requests
import feedparser
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
import os

# Configure logging at the start
logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

# Check for SPARQLStore availability
try:
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    HAS_SPARQLSTORE = True
except ImportError:
    HAS_SPARQLSTORE = False
    logger.warning("SPARQLStore not available. SPARQL Endpoint Mode and external queries will be disabled.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="RDF Explorer MCP Server v1.0.0")
parser.add_argument("--triple-file", default="", help="Path to the local RDF triple file")
parser.add_argument("--sparql-endpoint", default="", help="SPARQL endpoint URL (empty for Local File Mode)")
args = parser.parse_args()

logger.info("Starting RDF Explorer MCP Server v1.0.0")
logger.info("Setting lifespan")

# Define MCP instance as a global to ensure it's accessible
mcp = FastMCP(
    "RDF Explorer",
    dependencies=["rdflib[sparql]", "requests", "feedparser", "tiktoken"],
    lifespan=lambda mcp: triplestore_lifespan(mcp, args.triple_file, args.sparql_endpoint)
)

@asynccontextmanager
async def triplestore_lifespan(server: FastMCP, triple_file: str, sparql_endpoint: str) -> AsyncIterator[Dict[str, Any]]:
    """Manage the lifespan of the triplestore, initializing and shutting down the graph connection.

    Args:
        server (FastMCP): The FastMCP server instance.
        triple_file (str): Path to the local RDF triple file.
        sparql_endpoint (str): URL of the SPARQL endpoint, if any.

    Yields:
        Dict[str, Any]: Context dictionary containing the graph, metrics, and other state.

    Raises:
        FileNotFoundError: If the triple file is not found.
        Exception: If connecting to the SPARQL endpoint or parsing the file fails.
    """
    logger.info(f"Initializing triplestore with triple_file={triple_file}, sparql_endpoint={sparql_endpoint}")
    
    metrics = {"queries": 0, "total_time": 0.0}
    external_stores = {}
    feed_graph = rdflib.Graph()
    active_external_endpoint = None
    max_tokens = 10000
    
    if sparql_endpoint and HAS_SPARQLSTORE:
        logger.info(f"Connecting to SPARQL endpoint: {sparql_endpoint}")
        try:
            graph = SPARQLStore(query_endpoint=sparql_endpoint)
            graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
            external_stores[sparql_endpoint] = graph
            active_external_endpoint = sparql_endpoint
            logger.info(f"Successfully connected to {sparql_endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to SPARQL endpoint: {str(e)}")
            raise
    else:
        graph = rdflib.Graph()
        file_path = os.path.join(os.path.dirname(__file__), triple_file)
        logger.info(f"Loading local RDF file: {file_path}")
        try:
            graph.parse(file_path, format="turtle")
            logger.info(f"Loaded {len(graph)} triples from local file")
        except FileNotFoundError:
            logger.error(f"RDF file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load RDF file: {str(e)}")
            raise
    
    try:
        logger.info("Triplestore initialized successfully")
        yield {
            "graph": graph,
            "metrics": metrics,
            "external_stores": external_stores,
            "feed_graph": feed_graph,
            "active_external_endpoint": active_external_endpoint,
            "max_tokens": max_tokens,
            "triple_file": triple_file,
            "sparql_endpoint": sparql_endpoint
        }
    finally:
        logger.info("Shutting down triplestore connection")
        if sparql_endpoint and HAS_SPARQLSTORE and sparql_endpoint in external_stores:
            external_stores[sparql_endpoint].close()

# Resources
@mcp.resource("graph://{graph_id}")
def get_graph(graph_id: str) -> str:
    """Retrieve a graph by ID and serialize it in Turtle format.

    Args:
        graph_id (str): Identifier for the graph (currently unused, returns main graph).

    Returns:
        str: The serialized graph in Turtle format.

    Raises:
        Exception: If serialization fails.
    """
    logger.debug(f"Fetching graph for graph_id: {graph_id}")
    graph = mcp._lifespan_context["graph"]
    try:
        if HAS_SPARQLSTORE and isinstance(graph, rdflib.SPARQLStore):
            results = graph.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100")
            temp_graph = rdflib.Graph()
            for s, p, o in results:
                temp_graph.add((s, p, o))
            return temp_graph.serialize(format="turtle")
        return graph.serialize(format="turtle")
    except Exception as e:
        logger.error(f"Error serializing graph {graph_id}: {str(e)}")
        raise

@mcp.resource("feed://all")
def get_feed_graph() -> str:
    """Retrieve the feed graph stored by explore_url in Turtle format.

    Returns:
        str: The serialized feed graph in Turtle format.

    Raises:
        Exception: If serialization of the feed graph fails.
    """
    logger.debug("Fetching feed graph")
    feed_graph = mcp._lifespan_context["feed_graph"]
    try:
        return feed_graph.serialize(format="turtle")
    except Exception as e:
        logger.error(f"Error serializing feed graph: {str(e)}")
        raise


@mcp.resource("schema://all")
def get_schema() -> str:
    """Retrieve schema information (classes and properties) from the graph.
    Returns:
        str: A newline-separated string of schema elements (classes and properties).
    Raises:
        Exception: If the schema query fails.
    """
    graph = mcp._lifespan_context["graph"]
    schema_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT DISTINCT ?entity ?type
    WHERE {
        { ?entity a rdfs:Class . BIND("rdfs:Class" as ?type) } UNION
        { ?entity a owl:Class . BIND("owl:Class" as ?type) } UNION  
        { ?entity a rdf:Property . BIND("rdf:Property" as ?type) } UNION
        { ?entity a owl:ObjectProperty . BIND("owl:ObjectProperty" as ?type) } UNION
        { ?entity a owl:DatatypeProperty . BIND("owl:DatatypeProperty" as ?type) } UNION
        { ?entity a owl:AnnotationProperty . BIND("owl:AnnotationProperty" as ?type) }
    } 
    ORDER BY ?type ?entity
    LIMIT 100
    """
    try:
        results = graph.query(schema_query)
        return "\n".join(f"{row['type']}: {row['entity']}" for row in results)
    except Exception as e:
        logger.error(f"Schema query error: {str(e)}")
        raise


@mcp.resource("queries://{template_name}")
def get_query_template(template_name: str) -> str:
    """Retrieve a predefined SPARQL query template by name.

    Args:
        template_name (str): The name of the query template (e.g., 'orphans', 'cycles').

    Returns:
        str: The SPARQL query string or 'Template not found' if the name is invalid.
    """
    templates = {
        "orphans": "SELECT ?s WHERE { ?s ?p ?o . FILTER NOT EXISTS { ?x ?y ?s } } LIMIT 100",
        "cycles": "SELECT ?s ?o WHERE { ?s ?p ?o . ?o ?q ?s } LIMIT 100"
    }
    return templates.get(template_name, "Template not found")

@mcp.resource("explore://{query_name}")
def exploratory_query(query_name: str) -> str:
    """Execute an exploratory SPARQL query by name and return results in JSON.

    Args:
        query_name (str): The name of the exploratory query (e.g., 'classes', 'relationships/URI').

    Returns:
        str: JSON string of query results.

    Raises:
        Exception: If the query execution fails.
    """
    graph = mcp._lifespan_context["graph"]
    queries = {
        "classes": "SELECT DISTINCT ?type ?label WHERE { ?s a ?type . OPTIONAL { ?type rdfs:label ?label } } LIMIT 100",
        "properties": "SELECT DISTINCT ?objprop ?label WHERE { ?objprop a owl:ObjectProperty . OPTIONAL { ?objprop rdfs:label ?label } } LIMIT 100",
        "data_properties": "SELECT DISTINCT ?dataprop ?label WHERE { ?dataprop a owl:DatatypeProperty . OPTIONAL { ?dataprop rdfs:label ?label } } LIMIT 100",
        "used_properties": "SELECT DISTINCT ?p ?label WHERE { ?s ?p ?o . OPTIONAL { ?p rdfs:label ?label } } LIMIT 100",
        "entities": "SELECT DISTINCT ?entity ?elabel ?type ?tlabel WHERE { ?entity a ?type . OPTIONAL { ?entity rdfs:label ?elabel } . OPTIONAL { ?type rdfs:label ?tlabel } } LIMIT 100",
        "top_predicates": "SELECT ?pred (COUNT(*) as ?triples) WHERE { ?s ?pred ?o } GROUP BY ?pred ORDER BY DESC(?triples) LIMIT 100",
        "class_counts": "SELECT ?class (COUNT(?s) AS ?count) WHERE { ?s a ?class } GROUP BY ?class ORDER BY ?count LIMIT 100",
        "property_counts": "SELECT ?p (COUNT(?s) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p ORDER BY ?count LIMIT 100"
    }
    if query_name.startswith("relationships/"):
        subject = query_name.split("/", 1)[1]
        query = f"SELECT ?predicate ?object WHERE {{ <{subject}> ?predicate ?object }} LIMIT 100"
    else:
        query = queries.get(query_name, "Query not found")
    try:
        results = graph.query(query)
        return json.dumps([dict(row) for row in results])
    except Exception as e:
        logger.error(f"Exploratory query error: {str(e)}")
        raise

@mcp.resource("explore://report")
def exploratory_report() -> str:
    """Generate a Markdown report of exploratory queries.

    Returns:
        str: A Markdown-formatted report string.

    Raises:
        Exception: If any query in the report generation fails (error included in report).
    """
    graph = mcp._lifespan_context["graph"]
    report = ["# RDF Exploration Report"]
    for name in ["classes", "used_properties", "top_predicates"]:
        try:
            results = graph.query(mcp.call_resource(f"explore://{name}"))
            report.append(f"## {name.replace('_', ' ').title()}")
            report.append("| " + " | ".join(results.vars) + " |")
            report.append("| " + " | ".join(["---"] * len(results.vars)) + " |")
            for row in results:
                report.append("| " + " | ".join(str(row[var]) for var in results.vars) + " |")
        except Exception as e:
            report.append(f"## {name.replace('_', ' ').title()}\nError: {str(e)}")
    return "\n".join(report)

@mcp.resource("metrics://status")
def get_metrics() -> str:
    """Retrieve server metrics in JSON format.

    Returns:
        str: JSON string containing query count and total execution time.
    """
    metrics = mcp._lifespan_context["metrics"]
    return json.dumps(metrics)

# Tools
@mcp.tool()
def set_max_tokens(tokens: int, ctx: Context) -> str:
    """Set the maximum token limit for prompts.

    Args:
        tokens (int): The new maximum token limit (must be positive).
        ctx (Context): The FastMCP context object.

    Returns:
        str: Confirmation message or error if the value is invalid.
    """
    if tokens <= 0:
        return "Error: MAX_TOKENS must be positive."
    ctx.request_context.lifespan_context["max_tokens"] = tokens
    logger.info(f"Set MAX_TOKENS to {tokens}")
    return f"MAX_TOKENS set to {tokens}"

@mcp.tool()
def execute_on_endpoint(endpoint: str, query: str, ctx: Context) -> str:
    """Execute a SPARQL query directly on an external endpoint.

    Args:
        endpoint (str): The SPARQL endpoint URL to query.
        query (str): The SPARQL query to execute.
        ctx (Context): The FastMCP context object.

    Returns:
        str: Query results as a newline-separated string, or an error message if SPARQLStore is unavailable or the query fails.
    """
    if not HAS_SPARQLSTORE:
        return "SPARQLStore not available. Cannot query external endpoints."
    try:
        store = rdflib.SPARQLStore(query_endpoint=endpoint)
        results = store.query(query)
        logger.debug(f"Executed query on endpoint {endpoint}: {query}")
        return "\n".join(str(row) for row in results)
    except Exception as e:
        logger.error(f"Direct endpoint query error: {str(e)}")
        return f"Query error: {str(e)}"

@mcp.tool()
def connect_external_triplestore(endpoint: str, ctx: Context) -> str:
    """Connect to an external SPARQL endpoint and optionally set it as active for local mode queries.

    Args:
        endpoint (str): The SPARQL endpoint URL to connect to.
        ctx (Context): The FastMCP context object.

    Returns:
        str: Connection status message.

    Raises:
        Exception: If connecting to the endpoint fails.
    """
    if not HAS_SPARQLSTORE:
        return "SPARQLStore not available. Cannot connect to external endpoints."
    try:
        store = rdflib.SPARQLStore(query_endpoint=endpoint)
        store.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
        ctx.request_context.lifespan_context["external_stores"][endpoint] = store
        if not ctx.request_context.lifespan_context["active_external_endpoint"]:
            ctx.request_context.lifespan_context["active_external_endpoint"] = endpoint
            logger.info(f"Set active external endpoint to {endpoint} for local mode")
            return f"Connected to {endpoint} and set as active endpoint for local mode queries"
        else:
            logger.info(f"Connected to {endpoint} but not set as active (SPARQL endpoint mode active)")
            return f"Connected to {endpoint} (use SERVICE clause manually in SPARQL endpoint mode)"
    except Exception as e:
        logger.error(f"External triplestore connection error: {str(e)}")
        raise

@mcp.tool()
def sparql_query(query: str, ctx: Context, use_service: bool = True) -> str:
    """Execute a SPARQL query on the current graph or active external endpoint.

    Args:
        query (str): The SPARQL query to execute.
        ctx (Context): The FastMCP context object.
        use_service (bool): Whether to use a SERVICE clause for federated queries in local mode (default: True).

    Returns:
        str: Query results as a newline-separated string, or an error message if the query fails.
    """
    graph = ctx.request_context.lifespan_context["graph"]
    active_external_endpoint = ctx.request_context.lifespan_context["active_external_endpoint"]
    start_time = time.time()
    try:
        if not active_external_endpoint and active_external_endpoint and use_service:
            wrapped_query = f"SELECT ?s WHERE {{ SERVICE <{active_external_endpoint}> {{ {query} }} }}"
            logger.debug(f"Executing federated query in local mode: {wrapped_query}")
            results = graph.query(wrapped_query)
        else:
            logger.debug(f"Executing query directly: {query}")
            results = graph.query(query)
        ctx.request_context.lifespan_context["metrics"]["queries"] += 1
        ctx.request_context.lifespan_context["metrics"]["total_time"] += time.time() - start_time
        return "\n".join(str(row) for row in results)
    except Exception as e:
        logger.error(f"SPARQL query error: {str(e)}")
        return f"Query error: {str(e)}"

@mcp.tool()
def explore_url(url: str, ctx: Context) -> str:
    """Extract triples from an RSS/OPML feed URL and store them in the feed graph.

    In local mode, also merges into the main graph.

    Args:
        url (str): The URL of the feed to explore (e.g., 'http://rss.cnn.com/rss/cnn_topstories.rss').
        ctx (Context): The FastMCP context object.

    Returns:
        str: A message indicating the number of entries added.

    Raises:
        Exception: If fetching or parsing the feed fails.
    """
    graph = ctx.request_context.lifespan_context["graph"]
    feed_graph = ctx.request_context.lifespan_context["feed_graph"]
    try:
        response = requests.get(url)
        feed = feedparser.parse(response.content)
        for entry in feed.entries[:5]:
            feed_graph.add((rdflib.URIRef(entry.link), rdflib.URIRef("http://example.org/title"), rdflib.Literal(entry.title)))
        if not (HAS_SPARQLSTORE and isinstance(graph, SPARQLStore)):
            for triple in feed_graph:
                graph.add(triple)
        return f"Added {len(feed.entries[:5])} entries from {url} to feed_graph"
    except Exception as e:
        logger.error(f"Explore URL error: {str(e)}")
        raise

@mcp.tool()
def graph_stats(ctx: Context) -> str:
    """Calculate and return statistics about the graph in JSON format.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        str: JSON string containing graph statistics (e.g., triple count, unique subjects).

    Raises:
        Exception: If querying or calculating stats fails.
    """
    graph = ctx.request_context.lifespan_context["graph"]
    try:
        if HAS_SPARQLSTORE and isinstance(graph, rdflib.SPARQLStore):
            stats = {
                "unique_subjects": len(set(graph.query("SELECT DISTINCT ?s WHERE { ?s ?p ?o } LIMIT 1000"))),
                "unique_predicates": len(set(graph.query("SELECT DISTINCT ?p WHERE { ?s ?p ?o } LIMIT 1000"))),
                "unique_objects": len(set(graph.query("SELECT DISTINCT ?o WHERE { ?s ?p ?o } LIMIT 1000"))),
                "class_freq": dict(graph.query("SELECT ?class (COUNT(?s) AS ?count) WHERE { ?s a ?class } GROUP BY ?class LIMIT 100"))
            }
        else:
            stats = {
                "triple_count": len(graph),
                "unique_subjects": len(set(s for s, _, _ in graph)),
                "unique_predicates": len(set(p for _, p, _ in graph)),
                "unique_objects": len(set(o for _, _, o in graph)),
                "class_freq": dict(graph.query("SELECT ?class (COUNT(?s) AS ?count) WHERE { ?s a ?class } GROUP BY ?class"))
            }
        return json.dumps(stats)
    except Exception as e:
        logger.error(f"Graph stats error: {str(e)}")
        raise

@mcp.tool()
def count_triples(ctx: Context) -> str:
    """Count triples in the graph. Disabled in SPARQL Endpoint Mode; use a custom prompt instead.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        str: Number of triples as a string, or an error message if counting fails or in SPARQL mode.
    """
    if args.sparql_endpoint:  # Fixed: Use args instead of undefined SPARQL_ENDPOINT
        return "Error: count_triples is not supported in SPARQL Endpoint Mode. Write a custom SPARQL query to count triples."
    graph = ctx.request_context.lifespan_context["graph"]
    try:
        return str(len(graph))
    except Exception as e:
        return f"Error counting triples: {str(e)}"

@mcp.tool()
def full_text_search(search_term: str, ctx: Context) -> str:
    """Perform a full-text search on the graph or endpoint, avoiding proprietary syntax.

    Args:
        search_term (str): The term to search for.
        ctx (Context): The FastMCP context object.

    Returns:
        str: Search results as a newline-separated string, or an error message if the search fails.
    """
    graph = ctx.request_context.lifespan_context["graph"]
    query = f"""
    SELECT DISTINCT ?s ?label
    WHERE {{
      ?s ?p ?o .
      FILTER(REGEX(STR(?o), "{search_term}", "i"))
      OPTIONAL {{ ?s rdfs:label ?label }}
    }} LIMIT 100
    """
    try:
        results = graph.query(query)
        return "\n".join(str(row) for row in results)
    except Exception as e:
        logger.error(f"Full-text search error: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def health_check(ctx: Context) -> str:
    """Check the health of the triplestore connection.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        str: 'Healthy' if the connection is good, 'Unhealthy: <error>' otherwise.
    """
    graph = ctx.request_context.lifespan_context["graph"]
    try:
        graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
        return "Healthy"
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return f"Unhealthy: {str(e)}"

@mcp.tool()
def get_mode(ctx: Context) -> str:
    """Get the current mode of RDF Explorer. Useful for knowledge graph and semantic tech users to verify data source.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        str: A message indicating the mode and dataset or endpoint.
    """
    triple_file = ctx.request_context.lifespan_context["triple_file"]
    sparql_endpoint = ctx.request_context.lifespan_context["sparql_endpoint"]
    if sparql_endpoint:
        return f"SPARQL Endpoint Mode with Endpoint: '{sparql_endpoint}'"
    else:
        return f"Local File Mode with Dataset: '{triple_file}'"

# Prompts
@mcp.prompt()
def analyze_graph_structure(ctx: Context) -> list[base.Message]:
    """Initiate an analysis of the graph structure with sample schema data.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        list[base.Message]: A list of messages to guide graph structure analysis.

    Raises:
        Exception: If retrieving the schema fails.
    """
    try:
        schema = get_schema()
        source = "DBpedia" if ctx.request_context.lifespan_context["active_external_endpoint"] else "local triples"
        return [
            base.UserMessage(f"Please analyze the structure of the {source} graph."),
            base.UserMessage(f"Here's a sample schema:\n{schema}"),
            base.AssistantMessage("What specific aspects would you like me to focus on?")
        ]
    except Exception as e:
        logger.error(f"Analyze graph structure error: {str(e)}")
        raise

@mcp.prompt()
def find_relationships(subject: str) -> str:
    """Generate a SPARQL query to find relationships for a given subject.

    Args:
        subject (str): The URI of the subject to query relationships for.

    Returns:
        str: A SPARQL query string to find relationships.
    """
    return f"""
    Using the SPARQL query tool, find all relationships for the subject <{subject}>:
    SELECT ?predicate ?object WHERE {{ <{subject}> ?predicate ?object }} LIMIT 100
    """

@mcp.prompt()
def graph_visualization(subject: str) -> list[base.Message]:
    """Generate a DOT visualization of the graph around a subject.

    Args:
        subject (str): The URI of the subject to visualize.

    Returns:
        list[base.Message]: A list of messages containing the DOT graph.

    Raises:
        Exception: If querying the graph for visualization fails.
    """
    graph = mcp._lifespan_context["graph"]
    try:
        dot = ["digraph G {"]
        results = graph.query(f"SELECT ?p ?o WHERE {{ <{subject}> ?p ?o }} LIMIT 50")
        for row in results:
            dot.append(f'"{subject}" -> "{row["o"]}" [label="{row["p"]}"];')
        dot.append("}")
        return [
            base.UserMessage(f"Visualize the graph around <{subject}>"),
            base.AssistantMessage("\n".join(dot) + "\n\nUse Graphviz (dot -Tpng) to render this DOT format.")
        ]
    except Exception as e:
        logger.error(f"Graph visualization error: {str(e)}")
        raise

@mcp.prompt()
def text_to_sparql(prompt: str, ctx: Context) -> str:
    """Convert a text prompt to a SPARQL query and execute it, with token limit checks.

    Args:
        prompt (str): The text prompt to convert to SPARQL.
        ctx (Context): The FastMCP context object.

    Returns:
        str: Query results with usage stats, or an error message if execution fails or token limits are exceeded.
    """
    encoder = tiktoken.get_encoding("gpt2")
    start_time = time.time()
    grok_response = {"endpoint": None, "query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"}  # Placeholder
    endpoint = grok_response.get("endpoint")
    query = grok_response["query"]
    logger.debug(f"Prompt received: {prompt}")
    input_tokens = len(encoder.encode(prompt + query))
    max_tokens = ctx.request_context.lifespan_context["max_tokens"]
    if input_tokens > max_tokens:
        logger.debug(f"Token limit exceeded: {input_tokens} > {max_tokens}")
        return f"Error: Input exceeds token limit ({input_tokens} tokens > {max_tokens}). Shorten your prompt or increase MAX_TOKENS with 'set_max_tokens'."
    active_endpoint = ctx.request_context.lifespan_context["active_external_endpoint"]
    use_local = active_endpoint is None and endpoint is None
    use_configured = active_endpoint and (endpoint is None or endpoint == active_endpoint)
    use_extracted = endpoint and endpoint != active_endpoint
    logger.debug(f"Execution context - Local: {use_local}, Configured: {use_configured}, Extracted: {use_extracted}")
    try:
        if use_extracted:
            results = ctx.request_context.call_tool("execute_on_endpoint", {"endpoint": endpoint, "query": query})
            logger.debug(f"Executed on extracted endpoint {endpoint}")
        elif use_local:
            results = ctx.request_context.call_tool("sparql_query", {"query": query, "use_service": False})
            logger.debug("Executed on local graph")
        elif use_configured:
            results = ctx.request_context.call_tool("sparql_query", {"query": query})
            logger.debug(f"Executed on configured endpoint {active_endpoint}")
        else:
            logger.debug("No valid execution context")
            return "Unable to determine execution context for the query."
        output_tokens = len(encoder.encode(results))
        total_tokens = input_tokens + output_tokens
        exec_time = time.time() - start_time
        usage_stats = f"[Resource Usage: Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total: {total_tokens}, Time: {exec_time:.2f}s]"
        logger.debug(f"Usage stats generated: {usage_stats}")
        return f"{results}\n\n{usage_stats}"
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        if "interrupted" in str(e).lower():
            return f"Error: Response interrupted, likely due to token limit (Input: {input_tokens} tokens, Max: {max_tokens}). Shorten input or increase MAX_TOKENS."
        return f"Error executing query: {str(e)}"

@mcp.prompt()
def tutorial(ctx: Context) -> list[base.Message]:
    """Provide an interactive tutorial for RDF/SPARQL usage.

    Args:
        ctx (Context): The FastMCP context object.

    Returns:
        list[base.Message]: A list of tutorial messages tailored to the current mode.
    """
    source = "DBpedia" if args.sparql_endpoint else "local triples"  # Fixed: Use args
    example_query = "SELECT ?s WHERE { ?s a dbo:Person } LIMIT 10" if args.sparql_endpoint else "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
    example_viz = "http://dbpedia.org/resource/Albert_Einstein" if args.sparql_endpoint else "http://example.org/person1"
    return [
        base.UserMessage("Start the RDF/SPARQL tutorial"),
        base.AssistantMessage(f"Step 1: This uses {source}. Try 'explore://classes' to see types."),
        base.AssistantMessage(f"Step 2: Query with SPARQL. Try 'sparql_query' with '{example_query}'."),
        base.AssistantMessage(f"Step 3: Visualize with 'graph_visualization({example_viz})'. Ready for more?")
    ]

# Run the server
if __name__ == "__main__":
    logger.info("Starting mcp.run()")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start RDF Explorer: {str(e)}")
        sys.exit(1)
    logger.info("mcp.run() completed")