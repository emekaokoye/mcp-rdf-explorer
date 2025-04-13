# RDF Explorer v1.0.0 

## Overview
A Model Context Protocol (MCP) server that provides conversational interface for the exploration and analysis of RDF (Turtle) based Knowledge Graph in Local File mode or SPARQL Endpoint mode. This server facilitates communication between AI applications (hosts/clients) and RDF data, making graph exploration and analyzing graph data through SPARQL queries. A perfect tool for knowledge graph research and AI data preparation. 


## Components

### Tools
The server implements SPARQL queries and search functionality:

- `execute_on_endpoint`
   - Execute a SPARQL query directly on an external endpoint
   - Input:
     - `endpoint` (str): The SPARQL endpoint URL to query.
     - `query` (str): The SPARQL query to execute.
     - `ctx` (Context): The FastMCP context object.
   - Returns: Query results as a newline-separated string, or an error message.

- `sparql_query`
   - Execute a SPARQL query on the current graph or active external endpoint
   - Input:
     - `query` (str): The SPARQL query to execute.
     - `ctx` (Context): The FastMCP context object.
     - `use_service` (bool): Whether to use a SERVICE clause for federated queries in local mode (default: True).
   - Returns: Query results as a newline-separated string, or an error message.

- `graph_stats`
   - Calculate and return statistics about the graph in JSON format
   - Input:
     - `ctx` (Context): The FastMCP context object.
   - Returns: JSON string containing graph statistics (e.g., triple count, unique subjects).

- `count_triples`
   - Count triples in the graph. Disabled in SPARQL Endpoint Mode; use a custom prompt instead.
   - Input:
     - `ctx` (Context): The FastMCP context object.
   - Returns: Number of triples as a string, or an error message.


- `full_text_search`
   - Perform a full-text search on the graph or endpoint, avoiding proprietary syntax.
   - Input:
     - `search_term` (str): The term to search for.
     - `ctx` (Context): The FastMCP context object.
   - Returns: Search results as a newline-separated string, or an error message.


- `health_check`
   - Check the health of the triplestore connection.
   - Input:
     - `ctx` (Context): The FastMCP context object.
   - Returns: 'Healthy' if the connection is good, 'Unhealthy: <error>' otherwise.


- `get_mode`
   - Get the current mode of RDF Explorer. Useful for knowledge graph and semantic tech users to verify data source.
   - Input:
     - `ctx` (Context): The FastMCP context object.
   - Returns: A message indicating the mode and dataset or endpoint.


### Resources

The server exposes the following resources:
- `schema://all`: Retrieve schema information (classes and properties) from the graph.
  - Returns: A newline-separated string of schema elements (classes and properties).

- `queries://{template_name}`: Retrieve a predefined SPARQL query template by name.
  - Returns: The SPARQL query string or 'Template not found'.

- `explore://{query_name}`: Execute an exploratory SPARQL query by name and return results in JSON.
  - `query_name` (str): The name of the exploratory query (e.g., 'classes', 'relationships/URI').
  - Returns: JSON string of query results.

- `explore://report`: Generate a Markdown report of exploratory queries.
  - Returns: A Markdown-formatted report string.



### Prompts

The server exposes the following prompts:
- `analyze_graph_structure`: Initiate an analysis of the graph structure with schema data.
  - Returns: A list of messages to guide graph structure analysis.

- `find_relationships`: Generate a SPARQL query to find relationships for a given subject.
  - Returns: A SPARQL query string to find relationships.

- `text_to_sparql`: Convert a text prompt to a SPARQL query and execute it, with token limit checks.
  - `prompt` (str): The text prompt to convert to SPARQL.
  - Returns: Query results with usage stats, or an error message.
 



## Setup

## Configuration

### Installing on Claude Desktop
Before starting make sure [Node.js](https://nodejs.org/) is installed on your desktop for `npx` to work.
1. Go to: Settings > Developer > Edit Config

2. Add the following to your `claude_desktop_config.json`:
On MacOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

To use with a local RDF Turtle file, use this version with `--triple-file` args
```json
{
  "mcpServers": {
    "rdf_explorer": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\server.py", "--triple-file", "your_file.ttl"]
    }
  }
}
```

To use with a SPARQL Endpoint, use this version with `--sparql-endpoint` args
```json
{
  "mcpServers": {
    "rdf_explorer": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\server.py", "--sparql-endpoint", "https://example.com/sparql"]
    }
  }
}
```

3. Restart Claude Desktop and start querying and exploring graph data.

4. Prompt: "what mode is RDF Explorer running?"




## Usage Examples

Here are examples of how you can explore RDF data using natural language:

### Querying Data in Local File Mode

You can ask questions like:
- "Show me all employees in the Sales department"
- "Find the top 5 oldest customers"
- "Who has purchased more than 3 products in the last month?"
- "List all entities" 
- "Using the DBpedia endpoint, list 10 songs by Michael Jackson" 
- "Using the Wikidata endpoint, list 5 cities"
- "count the triples"
- "analyze the graph structure"
- "Select ..."
- "search '{text}' "
- "find relationships of '{URI}'"
- "what mode is RDF Explorer running?"

### Querying Data in SPARQL Endpoint Mode

You can ask questions like:
- "Using the DBpedia endpoint, list 10 songs by Michael Jackson" 
- "Using the Wikidata endpoint, list 5 cities"
- "Select ..."
- "search '{text}' "
- "find relationships of '{URI}'"
- "what mode is RDF Explorer running?"


## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the see the [LICENSE](LICENSE) file for details LICENSE file in the project repository.

