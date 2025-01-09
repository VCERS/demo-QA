#!/usr/bin/python3

from .prompts import search_prompt
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

host = config.get('General', 'host')
password = config.get('General', 'password')
user = config.get('General', 'user')
db = config.get('General', 'db')

enhanced_graph  = Neo4jGraph(url=host, username=user, password=password, database=db, enhanced_schema=True,)


from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# LSnPS
MATCH (a:ELT {{name: "LSnPS"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
      minLevel: 0,
      maxLevel: 100
  }})
    YIELD nodes, relationships
    MATCH (b:SOP) WHERE b IN nodes WITH b, nodes, relationships, collect(DISTINCT split(b.ceid, " ")[0]) AS ids
    MATCH (c:NUM) WHERE c IN nodes AND NOT split(c.ceid, " ")[0] in ids WITH collect(c) AS blacklist
    MATCH (a:ELT {{name: "LSnPS"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
    minLevel: 0,
    maxLevel: 100,
    blacklistNodes: blacklist
  }})
    YIELD nodes, relationships
    RETURN nodes, relationships;

# Li6PS5Cl
MATCH (a:ELT {{name: "Li6PS5Cl"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
      minLevel: 0,
      maxLevel: 100
  }})
    YIELD nodes, relationships
    MATCH (b:SOP) WHERE b IN nodes WITH b, nodes, relationships, collect(DISTINCT split(b.ceid, " ")[0]) AS ids
    MATCH (c:NUM) WHERE c IN nodes AND NOT split(c.ceid, " ")[0] in ids WITH collect(c) AS blacklist
    MATCH (a:ELT {{name: "Li6PS5Cl"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
    minLevel: 0,
    maxLevel: 100,
    blacklistNodes: blacklist
  }})
    YIELD nodes, relationships
    RETURN nodes, relationships;

# LSPSCl
MATCH (a:ELT {{name: "LSPSCl"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
      minLevel: 0,
      maxLevel: 100
  }})
    YIELD nodes, relationships
    MATCH (b:SOP) WHERE b IN nodes WITH b, nodes, relationships, collect(DISTINCT split(b.ceid, " ")[0]) AS ids
    MATCH (c:NUM) WHERE c IN nodes AND NOT split(c.ceid, " ")[0] in ids WITH collect(c) AS blacklist
    MATCH (a:ELT {{name: "LSPSCl"}})
    CALL apoc.path.subgraphAll(a, {{
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
    minLevel: 0,
    maxLevel: 100,
    blacklistNodes: blacklist
  }})
    YIELD nodes, relationships
    RETURN nodes, relationships;

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

QUESTION_ANSWER_TEMPLATE = """Task:Answer Questions based on previous information.
Instructions:
Do not simulate/hypothesize/assume anything.
Do not use any information that are not provided.
Format the answer using space. Bold all the titles.

Note: Do not include any explanations or apologies in your responses.
Also include Device, Precusor, Participants using the previous information if the question is about Synthesis path.
You can treat the two nodes that has the relationship Instance_of as the SAME node.
Examples: Here are a few examples of generated answer for particular questions:
# What is the synthesis path for LSnPS?
DEVICE: 
    ZrO2 ball-milling tank (1), 
    ZrO2 milling balls (25), 
    planetary ball miller (1), 
    a glovebox (1), 
    quartz tube (1), 
    muffle furnace (1), 
    the furnace (1), 
    Ar with attributes (O2 ≤ 0.1 ppm, H2O ≤ 0.1 ppm)

PRECURSOR: 
    Li2S with attributes (stoichiometric ratio, >99.9%), 
    P2S5 with attributes (stoichiometric ratio, 99%), 
    other raw material powders with attribute (stoichiometric ratio)

PARTICIPANTS: 
    None

SYNTHESIS PATHS: 
    Path 1:
        1. Mix Li2S, P2S5, and other raw material powders in a glovebox with Ar.
        2. Mill the mixture using ZrO2 ball-milling tank, ZrO2 milling balls, and planetary ball miller in a glovebox with Ar.
        3. Ball-mill the mixture in a glovebox with Ar to generate sulfide SE precursor.
        4. Grind the precursor to fine powders in a glovebox with Ar.
        5. Seal the powders into a quartz tube in a glovebox with Ar.
        6. Sinter the sealed tube using a muffle furnace.
        7. Cool down the tube to room temperature using the furnace in a glovebox with Ar, generating Li7P3S11 (The sulfide SEs).

The question is:
{question}"""

QUESTION_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question"], template=QUESTION_ANSWER_TEMPLATE
)



class KGSearchNeo(object):
  def __init__(self, llm):
    self.chain = GraphCypherQAChain.from_llm(
    llm,
    graph=enhanced_graph, 
    verbose=True, 
    allow_dangerous_requests=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT, 
    qa_prompt=QUESTION_ANSWER_PROMPT,
    )
  def extract(self, query: str):
    results = self.chain.invoke({'query': query})
    return results


